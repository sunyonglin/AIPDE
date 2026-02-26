import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
plt.rcParams.update({
    'font.family': ['SimHei', 'DejaVu Sans'],  # 中文字体优先，备用英文字体
    'axes.unicode_minus': False,  # 正确显示负号
    'mathtext.fontset': 'stix',  # 数学字体
    'figure.autolayout': True,  # 自动调整布局
})

# 参数设置
L = 40.0                     # 空间范围
N = 512                      # 空间格点数
dx = L / N                   # 空间步长
x = np.linspace(0, L, N, endpoint=False)  # 空间格点 [0, L)

dt = 0.01                    # 时间步长
T_total = 10.0               # 总时间
Nt = int(T_total / dt)       # 时间步数

# 亮孤子参数
eta = 1.0                    # 孤子振幅
k = 0.5                      # 波数，决定速度 v = k

# 计算频率 ω
omega = (k**2 - eta**2) / 2.0

# 1. 行波假设法（解析解）
def exact_solution(x, t, eta=eta, k=k, omega=omega, center_offset=0.0):
    """
    亮孤子精确解公式计算。
    ψ(x,t) = η * sech( η*((x - center_offset) - k*t) ) * exp( i*(k*x - ω*t) )
    注意：相位因子中的空间坐标仍为 x，以保证波数的物理意义。
    """
    # ξ 计算时考虑中心偏移，确保孤子初始位置在区域中部
    xi = (x - center_offset) - k * t
    envelope = eta / np.cosh(eta * xi)
    phase = np.exp(1j * (k * x - omega * t))
    return envelope * phase

# 2. 数值解法（Strang分裂法）
def strang_split(psi0, x, dt, Nt):
    """
    使用Strang分裂法数值求解NLSE（基于文档第3节）。
    psi0: 初始波函数（复数数组）
    x: 空间格点
    dt: 时间步长
    Nt: 时间步数
    返回: 每个时间步的波函数列表，共 Nt+1 个（包含初始状态）
    """
    N = len(x)
    # 修改点3：傅里叶变换的波数计算基于新的空间坐标范围 [0, L)
    # 对于周期区间长度为 L 的序列，fftfreq 的 d=dx 参数是正确的。
    k_vals = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)  # 波数数组

    psi_list = [psi0.copy()]
    psi_current = psi0.copy()

    for step in range(Nt):
        #半步势演化
        psi_current *= np.exp(1j * np.abs(psi_current)**2 * (dt/2))

        #整步动能演化
        psi_k = np.fft.fft(psi_current)                 # 傅里叶变换
        # 在傅里叶空间演化：∂ψ/∂t = - (i/2) * k^2 * ψ
        psi_k *= np.exp(-1j * (0.5 * k_vals**2) * dt)
        psi_current = np.fft.ifft(psi_k)                # 逆变换回实空间

        #另一个半步势演化
        psi_current *= np.exp(1j * np.abs(psi_current)**2 * (dt/2))

        psi_list.append(psi_current.copy())

    return psi_list

#区间[0, L]的中心为 L/2。
center = L / 2.0

# 初始条件：取精确解在 t=0 时的形式，并传入中心偏移量。
print("正在计算初始条件...")
psi0_exact = exact_solution(x, 0.0, center_offset=center)

# 数值求解
print("正在进行数值求解...")
psi_num_list = strang_split(psi0_exact, x, dt, Nt)

# 计算误差
print("计算误差...")
time_points = np.arange(0, Nt+1) * dt
errors = []

for i, t in enumerate(time_points):
    psi_exact = exact_solution(x, t, center_offset=center)
    psi_num = psi_num_list[i]

    # 计算 L2 范数误差（归一化）
    diff = np.abs(psi_exact - psi_num)
    error_L2 = np.linalg.norm(diff)
    errors.append(error_L2)

errors = np.array(errors)

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1：初始波函数模方（t=0）
ax = axes[0, 0]
ax.plot(x, np.abs(psi0_exact)**2, 'b-', label='解析解 |ψ|²', linewidth=2)
ax.plot(x, np.abs(psi_num_list[0])**2, 'r--', label='数值解 |ψ|²', linewidth=1.5)
ax.set_xlabel('位置 x')
ax.set_ylabel('|ψ(x,0)|²')
ax.set_title('t = 0 时的波包形状')
ax.legend()
ax.grid(True, alpha=0.3)
# 修改点6：设置x轴范围为新区间
ax.set_xlim([0, L])

# 图2：最终时刻波函数模方（t=T_total）
ax = axes[0, 1]
psi_exact_final = exact_solution(x, T_total, center_offset=center)
ax.plot(x, np.abs(psi_exact_final)**2, 'b-', label='解析解 |ψ|²', linewidth=2)
ax.plot(x, np.abs(psi_num_list[-1])**2, 'r--', label='数值解 |ψ|²', linewidth=1.5)
ax.set_xlabel('位置 x')
ax.set_ylabel(f'|ψ(x,{T_total})|²')
ax.set_title(f't = {T_total} 时的波包形状')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, L])

# 图3：时间误差图（主要结果）
ax = axes[1, 0]
ax.plot(time_points, errors, 'k-', linewidth=2)
ax.set_xlabel('时间 t')
ax.set_ylabel('L2误差')
ax.set_title('数值解与解析解之间的时间演化误差')
ax.grid(True, alpha=0.3)

# 图4：误差增长趋势（可选：显示误差增长率）
ax = axes[1, 1]
if len(errors) > 1:
    # 计算近似误差导数（数值差分）
    error_deriv = np.gradient(errors, time_points)
    ax.plot(time_points[:-1], error_deriv[:-1], 'g-', linewidth=1.5)
    ax.set_xlabel('时间 t')
    ax.set_ylabel('误差变化率 d(Error)/dt')
    ax.set_title('误差随时间的变化率')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'非线性薛定谔方程亮孤子解对比 (η={eta}, k={k}, dt={dt}, N={N})', fontsize=14)
plt.tight_layout()
plt.show()

# 输出关键误差信息
print(f"参数: 振幅 η = {eta}, 波数 k = {k}, 速度 v = k = {k}")
print(f"空间: 范围 [0, {L}], 格点数 N = {N}, 步长 dx = {dx:.4f}")
print(f"空间备注: 孤子初始中心位于 x = {center}")
print(f"时间: 总时长 T = {T_total}, 步长 dt = {dt}, 步数 Nt = {Nt}")
print(f"初始误差 (t=0): {errors[0]:.2e}")
print(f"最终误差 (t={T_total}): {errors[-1]:.2e}")

print(f"最大误差: {np.max(errors):.2e} (出现在 t ≈ {time_points[np.argmax(errors)]:.2f})")
