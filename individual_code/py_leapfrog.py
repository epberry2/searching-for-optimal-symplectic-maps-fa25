# leap_fold_leap_torch_noplot.py
# PyTorch rewrite of the provided TF v1 script, with ALL plotting removed.

import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------
# Helpers
# ----------------------------
def parse_kv_args(argv):
    d = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            d[k] = v
    return d

def to_device(x, device):
    return x.to(device) if torch.is_tensor(x) else torch.tensor(x, device=device)

def feature_poly_up_to_quintic(x: torch.Tensor, n: int) -> torch.Tensor:
    b = x.shape[0]
    x_i = x.view(b, n, 1)
    x_j = x.view(b, 1, n)
    quad = (x_i * x_j).reshape(b, n * n)
    cub = (quad.view(b, n * n, 1) * x.view(b, 1, n)).reshape(b, n ** 3)
    quart = (cub.view(b, n ** 3, 1) * x.view(b, 1, n)).reshape(b, n ** 4)
    quint = (quart.view(b, n ** 4, 1) * x.view(b, 1, n)).reshape(b, n ** 5)
    return torch.cat([x, quad, cub, quart, quint], dim=1)

class PolyScalar(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        feat_dim = n + n**2 + n**3 + n**4 + n**5
        self.weight = nn.Parameter(torch.zeros(feat_dim, dtype=torch.float32))
    def forward(self, x):
        feats = feature_poly_up_to_quintic(x, self.n)
        return feats @ self.weight

class FoldBlock(nn.Module):
    def __init__(self, n=2, depth=4, width=100, num_Lfg=1, activation=torch.tanh):
        super().__init__()
        assert n == 2
        self.n = n
        self.depth = depth
        self.width = width
        self.num_Lfg = num_Lfg
        self.activation = activation

        self.inputBiasL_x = nn.Parameter(torch.randn(num_Lfg, 1, width) * 0.1)
        self.inputML_x    = nn.Parameter(torch.randn(num_Lfg, 1, width) * 0.1)
        self.outputML_x   = nn.Parameter(torch.randn(num_Lfg, width, 2) * 0.1)
        
        self.inputBiasL_y = nn.Parameter(torch.randn(num_Lfg, 1, width) * 0.1)
        self.inputML_y    = nn.Parameter(torch.randn(num_Lfg, 1, width) * 0.1)
        self.outputML_y   = nn.Parameter(torch.randn(num_Lfg, width, 2) * 0.1)

        self.Lfg_neural_list_x = nn.ParameterList()
        self.Lfg_neural_list_y = nn.ParameterList()
        for _ in range(depth - 1):
            self.Lfg_neural_list_x.append(nn.Parameter(torch.randn(num_Lfg, width, width) * 0.1))
            self.Lfg_neural_list_x.append(nn.Parameter(torch.randn(num_Lfg, 1, width) * 0.1))
            self.Lfg_neural_list_y.append(nn.Parameter(torch.randn(num_Lfg, width, width) * 0.1))
            self.Lfg_neural_list_y.append(nn.Parameter(torch.randn(num_Lfg, 1, width) * 0.1))

        self.register_buffer("zero_row", torch.zeros(1, width))

    def _compute_L_single(self, inp, L_list, inputML, outputML, inputBiasL, i, txy: str):
        if txy == 'x':
            pad_first = torch.cat([inputML[i, :, :], self.zero_row], dim=0)
        else:
            pad_first = torch.cat([self.zero_row, inputML[i, :, :]], dim=0)
        h = torch.matmul(inp, pad_first) + inputBiasL[i, :, :]
        h = self.activation(h)
        for k in range(0, len(L_list), 2):
            W = L_list[k][i, :, :]
            B = L_list[k+1][i, :, :]
            h = self.activation(torch.matmul(h, W) + B)
        out = torch.matmul(h, outputML[i, :, :])
        return out

    def forward(self, x, y, dt):
        i = 0
        f_xy = self._compute_L_single(x, self.Lfg_neural_list_x, self.inputML_x, self.outputML_x, self.inputBiasL_x, i, 'x')
        g_yx = self._compute_L_single(y, self.Lfg_neural_list_y, self.inputML_y, self.outputML_y, self.inputBiasL_y, i, 'y')
        f_scalar = f_xy.sum(dim=1, keepdim=True)
        g_scalar = g_yx.sum(dim=1, keepdim=True)
        dLF_x = torch.autograd.grad(f_scalar.sum(), x, create_graph=True)[0]
        dLG_y = torch.autograd.grad(g_scalar.sum(), y, create_graph=True)[0]
        y_new = y + g_yx * dLF_x * dt
        x_new = x - f_xy * dLG_y * dt
        return x_new, y_new

class BlockParams(nn.Module):
    def __init__(self, n, num_macro_steps):
        super().__init__()
        self.V_list = nn.ModuleList([PolyScalar(n) for _ in range(num_macro_steps)])
        self.K_list = nn.ModuleList([PolyScalar(n) for _ in range(num_macro_steps)])
    def V(self, m, q): return self.V_list[m](q)
    def K(self, m, p): return self.K_list[m](p)

# ----------------------------
# Sampling domains
# ----------------------------
def ellipsoid_Eab_boundary(b, aa=1.0, bb=5.0, device="cpu"):
    z = torch.empty(b, 4, device=device).uniform_(-5, 5)
    q1, q2, p1, p2 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    z1sq = q1**2 + p1**2
    z2sq = q2**2 + p2**2
    F = math.pi * (z1sq / aa + z2sq / bb)
    div = torch.sqrt(F).unsqueeze(1)
    return z / div

def polydisk_boundary(b, aa=1.0, bb=5.0, device="cpu"):
    length = torch.sqrt(torch.empty(b, 2, device=device).uniform_(0, 1))
    angle  = math.pi * torch.empty(b, 2, device=device).uniform_(0, 2)
    half = b // 2
    length[:half, 0] = 1.0
    length[half:, 1] = 1.0
    q = length * torch.cos(angle)
    p = length * torch.sin(angle)
    z1 = torch.sqrt(torch.tensor(aa, device=device)) * torch.stack([q[:, 0], p[:, 0]], dim=1) / math.sqrt(math.pi)
    z2 = torch.sqrt(torch.tensor(bb, device=device)) * torch.stack([q[:, 1], p[:, 1]], dim=1) / math.sqrt(math.pi)
    out = torch.cat([z1, z2], dim=1)[:, [0, 2, 1, 3]]
    return out

# ----------------------------
# Training loop
# ----------------------------
def main():
    inp_args = parse_kv_args(sys.argv[1:])

    restore_session = False
    local_training_steps = int(inp_args.get("local_training_steps", 15000))
    current_lr = float(inp_args.get("current_lr", 0.0))
    decay_steps = int(inp_args.get("decay_steps", 20))
    decay_rate = float(inp_args.get("decay_rate", 1.0))
    num_steps_basic = int(inp_args.get("num_steps_basic", 1))
    num_macro_steps = int(inp_args.get("num_macro_steps", 15))
    b_basic = int(inp_args.get("b_basic", 5000))

    num_steps_check_accuracy = int(inp_args.get("num_steps_check_accuracy", num_steps_basic))
    b_check_accuracy = int(inp_args.get("b_check_accuracy", 10000))

    b_movie = int(inp_args.get("b_movie", b_basic))
    num_steps_movie = int(inp_args.get("num_steps_movie", 100))

    save_steps = int(inp_args.get("save_steps", 500))
    check_accuracy_steps = int(inp_args.get("check_accuracy_steps", 500))
    write_steps = int(inp_args.get("write_steps", 5))

    n = 2
    tot_secs = float(inp_args.get("tot_secs", 1.0))
    aa = float(inp_args.get("aa", 1.0))
    bb = float(inp_args.get("bb", 5.0))
    domain = str(inp_args.get("domain", "ellipsoid"))

    assert (local_training_steps * b_basic * 2 * n * 4) / (2.0**30) <= 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tic = time.time()

    blk1 = BlockParams(n=1, num_macro_steps=num_macro_steps).to(device)
    blk2 = BlockParams(n=2, num_macro_steps=num_macro_steps).to(device)
    blk3 = BlockParams(n=1, num_macro_steps=num_macro_steps).to(device)
    fold = FoldBlock(n=2, depth=4, width=100, num_Lfg=1, activation=torch.tanh).to(device)

    params = list(blk1.parameters()) + list(blk2.parameters()) + list(blk3.parameters()) + list(fold.parameters())
    opt = torch.optim.Adam(params, lr=current_lr)

    # TensorBoard
    if not os.path.isdir('summaries'):
        run_num = 0
        os.makedirs('summaries', exist_ok=True)
    else:
        folders = [f for f in os.listdir('./summaries') if f.startswith("run")]
        run_num = (max([int(f[3:]) for f in folders]) + 1) if folders else 0
    writer = SummaryWriter(log_dir=f"summaries/run{run_num}")
    writer.add_text('annotations', f"{domain} aa={aa} bb={bb}\nsplit quartic Hamiltonian")
    writer.add_scalar('num_steps_basic', num_steps_basic, 0)
    writer.add_scalar('num_macro_steps', num_macro_steps, 0)
    writer.add_scalar('b_basic', b_basic, 0)
    writer.add_scalar('num_steps_check_accuracy', num_steps_check_accuracy, 0)
    writer.add_scalar('b_check_accuracy', b_check_accuracy, 0)

    def get_z_init(b=None):
        if b is None:
            b = b_basic
        if domain == 'ellipsoid':
            return ellipsoid_Eab_boundary(b, aa=aa, bb=bb, device=device)
        elif domain == 'polydisk':
            return polydisk_boundary(b, aa=aa, bb=bb, device=device)
        else:
            raise RuntimeError("Domain not recognized")

    global_step = 0
    dt_basic = tot_secs / (num_macro_steps * float(num_steps_basic))

    for step in range(local_training_steps):
        z = get_z_init(b_basic).requires_grad_(True)
        q1 = z[:, 0:1]; q2 = z[:, 1:2]; p1 = z[:, 2:3]; p2 = z[:, 3:4]
        dt = to_device(dt_basic, device)

        # Block 1 + Fold (per macro)
        for m in range(num_macro_steps):
            K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=True)[0]
            q1 = q1 + dK_dp * 0.5 * dt
            for _ in range(num_steps_basic - 1):
                V = blk1.V(m, q1); dV_dq = torch.autograd.grad(V.sum(), q1, create_graph=True)[0]
                p1 = p1 - dV_dq * dt
                K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=True)[0]
                q1 = q1 + dK_dp * dt
            V = blk1.V(m, q1); dV_dq = torch.autograd.grad(V.sum(), q1, create_graph=True)[0]
            p1 = p1 - dV_dq * dt
            K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=True)[0]
            q1 = q1 + dK_dp * 0.5 * dt

            x_t = torch.cat([q1, q2], dim=1).requires_grad_(True)
            y_t = torch.cat([p1, p2], dim=1).requires_grad_(True)
            x_new, y_new = fold(x_t, y_t, dt)
            q1, q2 = x_new[:, 0:1], x_new[:, 1:2]
            p1, p2 = y_new[:, 0:1], y_new[:, 1:2]

        # Block 2
        q = torch.cat([q1, q2], dim=1).requires_grad_(True)
        p = torch.cat([p1, p2], dim=1).requires_grad_(True)
        for m in range(num_macro_steps):
            K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=True)[0]
            q = q + dK_dp * 0.5 * dt
            for _ in range(num_steps_basic - 1):
                V = blk2.V(m, q); dV_dq = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
                p = p - dV_dq * dt
                K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=True)[0]
                q = q + dK_dp * dt
            V = blk2.V(m, q); dV_dq = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
            p = p - dV_dq * dt
            K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=True)[0]
            q = q + dK_dp * 0.5 * dt

        # Block 3
        q1 = q[:, 0:1]; q2 = q[:, 1:2]; p1 = p[:, 0:1]; p2 = p[:, 1:2]
        for m in range(num_macro_steps):
            K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=True)[0]
            q2 = q2 + dK_dp * 0.5 * dt
            for _ in range(num_steps_basic - 1):
                V = blk3.V(m, q2); dV_dq = torch.autograd.grad(V.sum(), q2, create_graph=True)[0]
                p2 = p2 - dV_dq * dt
                K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=True)[0]
                q2 = q2 + dK_dp * dt
            V = blk3.V(m, q2); dV_dq = torch.autograd.grad(V.sum(), q2, create_graph=True)[0]
            p2 = p2 - dV_dq * dt
            K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=True)[0]
            q2 = q2 + dK_dp * 0.5 * dt

        last_z = torch.cat([q1, q2, p1, p2], dim=1)
        radius_sq = (last_z * last_z).sum(dim=1)
        enclosing_area = math.pi * torch.max(radius_sq)
        loss = enclosing_area

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        writer.add_scalar("enclosing_area", enclosing_area.item(), global_step)
        writer.add_scalar("loss", loss.item(), global_step)
        writer.add_scalar("lr", opt.param_groups[0]['lr'], global_step)

        if (step % save_steps) == (save_steps - 1):
            torch.save({
                "blk1": blk1.state_dict(),
                "blk2": blk2.state_dict(),
                "blk3": blk3.state_dict(),
                "fold": fold.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "run_num": run_num
            }, "./save_variables.pt")
            print("Saved variables.")

        if (step % write_steps) == (write_steps - 1):
            print("Wrote variables to tensorboard.")

        if (global_step % decay_steps) == (decay_steps - 1):
            new_lr = decay_rate * opt.param_groups[0]['lr']
            for g in opt.param_groups:
                g['lr'] = new_lr

        if (step % check_accuracy_steps) == (check_accuracy_steps - 1):
            # NOTE: evaluation needs grads for local updates -> don't wrap in no_grad()
            z_chk = get_z_init(b_check_accuracy).requires_grad_(True)
            q1 = z_chk[:, 0:1]; q2 = z_chk[:, 1:2]; p1 = z_chk[:, 2:3]; p2 = z_chk[:, 3:4]
            dtc = to_device(tot_secs / (num_macro_steps * float(num_steps_check_accuracy)), device)
            for m in range(num_macro_steps):
                K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=False)[0]
                q1 = q1 + dK_dp * 0.5 * dtc
                for _ in range(num_steps_check_accuracy - 1):
                    V = blk1.V(m, q1); dV_dq = torch.autograd.grad(V.sum(), q1, create_graph=False)[0]
                    p1 = p1 - dV_dq * dtc
                    K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=False)[0]
                    q1 = q1 + dK_dp * dtc
                V = blk1.V(m, q1); dV_dq = torch.autograd.grad(V.sum(), q1, create_graph=False)[0]
                p1 = p1 - dV_dq * dtc
                K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=False)[0]
                q1 = q1 + dK_dp * 0.5 * dtc

                x_t = torch.cat([q1, q2], dim=1).requires_grad_(True)
                y_t = torch.cat([p1, p2], dim=1).requires_grad_(True)
                x_new, y_new = fold(x_t, y_t, dtc)
                q1, q2 = x_new[:, 0:1], x_new[:, 1:2]
                p1, p2 = y_new[:, 0:1], y_new[:, 1:2]

            q = torch.cat([q1, q2], dim=1).requires_grad_(True)
            p = torch.cat([p1, p2], dim=1).requires_grad_(True)
            for m in range(num_macro_steps):
                K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=False)[0]
                q = q + dK_dp * 0.5 * dtc
                for _ in range(num_steps_check_accuracy - 1):
                    V = blk2.V(m, q); dV_dq = torch.autograd.grad(V.sum(), q, create_graph=False)[0]
                    p = p - dV_dq * dtc
                    K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=False)[0]
                    q = q + dK_dp * dtc
                V = blk2.V(m, q); dV_dq = torch.autograd.grad(V.sum(), q, create_graph=False)[0]
                p = p - dV_dq * dtc
                K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=False)[0]
                q = q + dK_dp * 0.5 * dtc

            q1 = q[:, 0:1]; q2 = q[:, 1:2]; p1 = p[:, 0:1]; p2 = p[:, 1:2]
            for m in range(num_macro_steps):
                K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=False)[0]
                q2 = q2 + dK_dp * 0.5 * dtc
                for _ in range(num_steps_check_accuracy - 1):
                    V = blk3.V(m, q2); dV_dq = torch.autograd.grad(V.sum(), q2, create_graph=False)[0]
                    p2 = p2 - dV_dq * dtc
                    K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=False)[0]
                    q2 = q2 + dK_dp * dtc
                V = blk3.V(m, q2); dV_dq = torch.autograd.grad(V.sum(), q2, create_graph=False)[0]
                p2 = p2 - dV_dq * dtc
                K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=False)[0]
                q2 = q2 + dK_dp * 0.5 * dtc

            last_chk = torch.cat([q1, q2, p1, p2], dim=1)
            more_acc = math.pi * torch.max((last_chk * last_chk).sum(dim=1))
            writer.add_scalar('more_accurate_enclosing_area', more_acc.item(), global_step)
            print(f'using num steps: {num_steps_check_accuracy}')
            print(f'enclosing area for b = {b_check_accuracy}:')
            print(more_acc.item())

        print(f'using num steps: {num_steps_basic}, local step: {step}, global step: {global_step}, '
              f'current_lr: {opt.param_groups[0]["lr"]:.6f}, loss: {loss.item():.6f}, '
              f'enclosing area: {enclosing_area.item():.6f}')
        global_step += 1

    torch.save({
        "blk1": blk1.state_dict(),
        "blk2": blk2.state_dict(),
        "blk3": blk3.state_dict(),
        "fold": fold.state_dict(),
        "opt": opt.state_dict(),
        "step": local_training_steps - 1,
        "run_num": run_num
    }, "./save_variables.pt")
    print("Saved variables.")

    # movie trajectory (kept; no plotting, only .npy save)
    z = get_z_init(b_movie).requires_grad_(True)
    z_traj = [z.detach().cpu().numpy()]
    dtm = to_device(tot_secs / (num_macro_steps * float(num_steps_movie)), device)
    q1 = z[:, 0:1]; q2 = z[:, 1:2]; p1 = z[:, 2:3]; p2 = z[:, 3:4]
    for m in range(num_macro_steps):
        K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=False)[0]
        q1 = q1 + dK_dp * 0.5 * dtm
        z_traj.append(torch.cat([q1, q2, p1, p2], dim=1).cpu().numpy())
        for _ in range(num_steps_movie - 1):
            V = blk1.V(m, q1); dV_dq = torch.autograd.grad(V.sum(), q1, create_graph=False)[0]
            p1 = p1 - dV_dq * dtm
            K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=False)[0]
            q1 = q1 + dK_dp * dtm
            z_traj.append(torch.cat([q1, q2, p1, p2], dim=1).cpu().numpy())
        V = blk1.V(m, q1); dV_dq = torch.autograd.grad(V.sum(), q1, create_graph=False)[0]
        p1 = p1 - dV_dq * dtm
        K = blk1.K(m, p1); dK_dp = torch.autograd.grad(K.sum(), p1, create_graph=False)[0]
        q1 = q1 + dK_dp * 0.5 * dtm
        z_traj.append(torch.cat([q1, q2, p1, p2], dim=1).cpu().numpy())
        x_t = torch.cat([q1, q2], dim=1).requires_grad_(True)
        y_t = torch.cat([p1, p2], dim=1).requires_grad_(True)
        x_new, y_new = fold(x_t, y_t, dtm)
        q1, q2 = x_new[:, 0:1], x_new[:, 1:2]
        p1, p2 = y_new[:, 0:1], y_new[:, 1:2]
        z_traj.append(torch.cat([q1, q2, p1, p2], dim=1).cpu().numpy())
    q = torch.cat([q1, q2], dim=1).requires_grad_(True)
    p = torch.cat([p1, p2], dim=1).requires_grad_(True)
    for m in range(num_macro_steps):
        K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=False)[0]
        q = q + dK_dp * 0.5 * dtm
        z_traj.append(torch.cat([q[:, 0:1], q[:, 1:2], p[:, 0:1], p[:, 1:2]], dim=1).cpu().numpy())
        for _ in range(num_steps_movie - 1):
            V = blk2.V(m, q); dV_dq = torch.autograd.grad(V.sum(), q, create_graph=False)[0]
            p = p - dV_dq * dtm
            K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=False)[0]
            q = q + dK_dp * dtm
            z_traj.append(torch.cat([q[:, 0:1], q[:, 1:2], p[:, 0:1], p[:, 1:2]], dim=1).cpu().numpy())
        V = blk2.V(m, q); dV_dq = torch.autograd.grad(V.sum(), q, create_graph=False)[0]
        p = p - dV_dq * dtm
        K = blk2.K(m, p); dK_dp = torch.autograd.grad(K.sum(), p, create_graph=False)[0]
        q = q + dK_dp * 0.5 * dtm
        z_traj.append(torch.cat([q[:, 0:1], q[:, 1:2], p[:, 0:1], p[:, 1:2]], dim=1).cpu().numpy())
    q1 = q[:, 0:1]; q2 = q[:, 1:2]; p1 = p[:, 0:1]; p2 = p[:, 1:2]
    for m in range(num_macro_steps):
        K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=False)[0]
        q2 = q2 + dK_dp * 0.5 * dtm
        z_traj.append(torch.cat([q1, q2, p1, p2], dim=1).cpu().numpy())
        for _ in range(num_steps_movie - 1):
            V = blk3.V(m, q2); dV_dq = torch.autograd.grad(V.sum(), q2, create_graph=False)[0]
            p2 = p2 - dV_dq * dtm
            K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=False)[0]
            q2 = q2 + dK_dp * dtm
            z_traj.append(torch.cat([q1, q2, p1, p2], dim=1).cpu().numpy())
        V = blk3.V(m, q2); dV_dq = torch.autograd.grad(V.sum(), q2, create_graph=False)[0]
        p2 = p2 - dV_dq * dtm
        K = blk3.K(m, p2); dK_dp = torch.autograd.grad(K.sum(), p2, create_graph=False)[0]
        q2 = q2 + dK_dp * 0.5 * dtm
        z_traj.append(torch.cat([q1, q2, p1, p2], dim=1).cpu().numpy())
    z_traj = np.stack(z_traj, axis=0)

    toc = time.time()
    print(f"Total time elapsed: {toc - tic:.2f}s")

    trj_name = f"trj_run{run_num}.npy"
    np.save(trj_name, z_traj)
    print(f"saved {trj_name} (flow for the time dependent Hamiltonian at end of training)")
    # NOTE: no training_plot_points_list / trn_run*.npy anymore

if __name__ == "__main__":
    main()
