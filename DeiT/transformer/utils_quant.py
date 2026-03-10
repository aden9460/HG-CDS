import torch
import torch.nn as nn


class BinaryQuantizer(torch.autograd.Function):
    """
    Binary quantizer - maps real values to {-1, +1}

    Paper formula: sign(x) applied in forward pass

    Backward pass uses Straight-Through Estimator (STE):
    Standard STE as defined in the paper:
        ∂sign(x)/∂x ≈ { 1,  if |x| ≤ 1
                       { 0,  otherwise

    This implementation uses improved gradient estimation with linear interpolation in [-1,1]:
        ∂sign(x)/∂x ≈ { 2 + 2x,  if -1 ≤ x ≤ 0
                       { 2 - 2x,  if 0 < x ≤ 1
                       { 0,       otherwise
    This gradient form peaks at 2 when x=0 and is 0 at boundaries, providing smoother gradient transition
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: apply sign function sign(x)

        Args:
            input: Input tensor (can be weight W or activation X)
        Returns:
            out: Binarized output ∈ {-1, +1}
        """
        ctx.save_for_backward(input)
        out = torch.sign(input)  # sign(x) = +1 if x>0, -1 if x<0, 0 if x=0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: approximate gradient using improved STE

        Standard STE has gradient 1 for |x|≤1; this implementation uses triangular gradient:
        - x∈[-1,0]: grad = 2 + 2x (linearly grows from 0 to 2)
        - x∈(0,1]:  grad = 2 - 2x (linearly decays from 2 to 0)
        - elsewhere: grad = 0

        Args:
            grad_output: Gradient from upstream
        Returns:
            grad_input: Gradient passed to downstream
        """
        input = ctx.saved_tensors
        input = input[0]
        # indicate_small = (input < -1).float()  # region < -1: gradient = 0
        # indicate_big = (input > 1).float()       # region > 1: gradient = 0

        # Left half [-1, 0]: gradient = 2 + 2x
        indicate_leftmid = ((input >= -1) & (input <= 0)).float()
        # Right half (0, 1]: gradient = 2 - 2x
        indicate_rightmid = ((input > 0) & (input <= 1)).float()

        # Apply piecewise linear gradient
        grad_input = (indicate_leftmid * (2 + 2*input) + indicate_rightmid * (2 - 2*input)) * grad_output.clone()
        return grad_input


class BiTBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val):
        ctx.save_for_backward(input, clip_val)
        out = torch.round(input / clip_val).clamp(0.0, 1.0) * clip_val
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        q_w = input / clip_val
        indicate_small = (q_w < 0.0).float()
        indicate_big = (q_w > 1.0).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)

        grad_clip_val = ((indicate_middle * (q_w.round() - q_w) + indicate_big) * grad_output).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output.clone()
        return grad_input, grad_clip_val


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeLinear(nn.Linear):
    """
    Quantized linear layer - implements binary matrix multiplication per paper Eq.(1)

    Paper Eq.(1):
        Y(X) = αW ⊙ sign(X) ⊗ sign(W - μ(W))

    其中:
        - X ∈ R^(N×Din): input activation matrix
        - W ∈ R^(Din×Dout): weight matrix
        - βX ∈ R^Din: activation threshold vector (learned via backprop)
        - μ(W) ∈ R^Dout: weight threshold (mean of each output column)
        - αW ∈ R: scaling factor, αW = (1/n)||W||₁
        - Rsign = sign(X + βX): threshold-based activation binarization
        - ⊗: popcount operation (binary matrix multiplication)

    Note: βX is implicitly handled by subsequent affine transform layer (move parameter)
    """
    def __init__(self,  *kargs, bias=False, config=None):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits

        # Select quantizer based on bit-width
        if self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer  # binary quantization
        elif self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer  # ternary quantization
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits < 32:
            self.weight_quantizer = SymQuantizer  # symmetric quantization
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

        if self.input_bits == 1:
            self.act_quantizer = BinaryQuantizer  # activation binarization
        elif self.input_bits == 2:
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.input_bits < 32:
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))


    def forward(self, input):
        """
        Forward pass - implements paper Eq.(1)

        For binary weights (weight_bits==1):
            1. Compute scaling factor: αW = (1/n)||W||₁ = mean(|W|) per output channel
            2. Compute threshold: μ(W) = mean(W) per output channel
            3. Binarize: sign(W - μ(W))
            4. Scale: αW * sign(W - μ(W))
            5. Gradient propagation via STE

        Args:
            input: activation X ∈ R^(N×Din)
        Returns:
            out: output Y ∈ R^(N×Dout)
        """
        if self.weight_bits == 1:
            # ===== Step 1: Compute scaling factor αW =====
            # αW = (1/n)||W||₁, averaged per output channel (dim=1 corresponds to Dout)
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()  # stop gradient from flowing to scaling factor

            # ===== Step 2: Compute threshold μ(W) and center weights =====
            # μ(W) = mean(W) per output channel (dim=-1 corresponds to Din)
            # real_weights = W - μ(W), implements W - μ(W) from the paper
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)

            # ===== Step 3: Binarize weights =====
            # sign(W - μ(W)), then multiply by scaling factor αW
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)

            # ===== Step 4: STE (Straight-Through Estimator) =====
            # Forward uses binary weights; backward uses clipped gradients
            # This trick: forward uses binary, backward uses clipped real
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        elif self.weight_bits < 32:
            # Multi-bit quantization
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            # Full precision
            weight = self.weight

        # ===== Activation quantization =====
        if self.input_bits == 1:
            # Apply sign to input: Rsign = sign(X + βX)
            # Note: βX threshold is learned in the subsequent move parameter
            input = self.act_quantizer.apply(input)

        # ===== Matrix multiplication ⊗ =====
        # For binary inputs and weights, this is an XNOR-popcount operation
        # PyTorch will automatically use an efficient implementation
        out = nn.functional.linear(input, weight)

        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeConv2d(nn.Conv2d):
    def __init__(self,  *kargs, bias=True, config=None):
        super(QuantizeConv2d, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits
        
        if self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        elif self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits < 32:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            
        if self.input_bits == 1:
            self.act_quantizer = BinaryQuantizer
        elif self.input_bits == 2:
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.input_bits < 32:
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
 

    def forward(self, input):
        if self.weight_bits == 1:
            # This forward pass is meant for only binary weights and activations
            real_weights = self.weight
            scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            #print(scaling_factor, flush=True)
            scaling_factor = scaling_factor.detach()
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
            #print(binary_weights, flush=True)
        elif self.weight_bits < 32:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight

        if self.input_bits == 1:
            input = self.act_quantizer.apply(input)
        
        out = nn.functional.conv2d(input, weight, stride=self.stride, padding=self.padding)
        
        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(self,  *kargs,padding_idx=None, config=None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.init = True
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, self.layerwise)
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out