import numpy as np
from scipy import integrate
from scipy.linalg import eigh
from scipy.sparse import diags, csr_matrix
import scipy.special as sp
import matplotlib.pyplot as plt

# 参数设置
L = 10.0  # 空间范围 [-L, L]
M = 512  # 空间格点数
dt = 0.001  # 时间步长
T_final = 1.0  # 总时间
N = int(T_final / dt)  # 时间步数

# 空间离散
dx = 2 * L / M
x = np.linspace(-L, L, M, endpoint=False)
time_grid = np.linspace(0, T_final, N)


#求解方程：i∂_tψ = -1/2∂_x^2ψ+1/2 ω(t)^2 x^2 ψ
# 谐振子频率函数（随时间变化）
def omega(t):
    return 1.0 + t / T_final


# 1.数值求解
print("开始数值求解")

# 构建动能矩阵（周期性边界条件）
main_diag = np.ones(M) * (-2.0) / dx ** 2
off_diag = np.ones(M - 1) / dx ** 2

# 构建三对角矩阵
P = diags([off_diag, main_diag, off_diag], [-1, 0, 1])
P = P.tolil()
# 处理周期性边界条件
P[0, M - 1] = 1.0 / dx ** 2
P[M - 1, 0] = 1.0 / dx ** 2
P = P.tocsr()
P = -0.5 * P  # 乘以系数 -1/2

# 初始化波函数为谐振子基态 (n=0)
psi = np.exp(-x**2 / 2)                     # 未归一化的基态
psi = psi / (np.linalg.norm(psi) * np.sqrt(dx))  # 离散归一化，保证 ∑|ψ|² Δx = 1

# 存储波函数演化
psi_history = np.zeros((N, M), dtype=complex)
psi_history[0] = psi

# 时间演化循环
for j in range(N - 1):
    t_mid = j * dt + dt / 2  # 时间中点
    omega_mid = omega(t_mid)

    # 构建势能矩阵（对角矩阵）
    V = 0.5 * omega_mid ** 2 * x ** 2
    V_matrix = diags([V], [0])

    # 构建离散哈密顿量 H_D = P + V_matrix
    H_D = P + V_matrix

    # 创建单位矩阵
    I_sparse = diags([np.ones(M)], [0])

    # 第一步：计算中间变量 |y> = (1 - i*dt/2 * H_D) |ψ^(j)>
    psi_j = psi_history[j]
    y = psi_j - 1j * dt / 2 * (H_D @ psi_j)

    # 第二步：直接求解 (1 + i*dt/2 * H_D) |ψ^(j+1)> = |y>
    # 构建系数矩阵 A = I + i*dt/2 * H_D
    A = I_sparse + 1j * dt / 2 * H_D

    # 将稀疏矩阵转换为稠密矩阵求解
    A_dense = A.toarray()

    # 求解线性方程组 A * psi_{j+1} = y
    psi_j1 = np.linalg.solve(A_dense, y)

    # 归一化
    norm = np.linalg.norm(psi_j1) * np.sqrt(dx)
    if norm > 0:
        psi_j1 = psi_j1 / norm

    # 更新波函数
    psi_history[j + 1] = psi_j1

    if (j + 1) % 100 == 0:
        norm_current = np.sum(np.abs(psi_j1) ** 2) * dx
        print(f"数值求解: 时间步 {j + 1}/{N - 1}, 模方: {norm_current:.6f}")

# 计算数值解模方
norm_numerical = np.zeros(N)
for j in range(N):
    norm_numerical[j] = np.sum(np.abs(psi_history[j]) ** 2) * dx

print(f"数值求解完成, 最终模方: {norm_numerical[-1]:.8f}")

# 2. 精确求解
print("\n开始精确求解")

# 文档参数设置
omega0 = 1.0
beta0 = 1.0
m = 2
k = m + 2
a = np.sqrt(omega0)


# 计算f(t)函数
def compute_f(t, T_final=T_final):
    T_half = 0.5 * T_final
    arg_squared = 0.5 * T_final * (1 + t / T_final) ** 2

    J_minus_1_4_T = sp.jv(-1 / 4, T_half)
    J_1_4_T = sp.jv(1 / 4, T_half)
    J_minus_5_4_T = sp.jv(-5 / 4, T_half)
    J_5_4_T = sp.jv(5 / 4, T_half)

    J_minus_1_4_arg = sp.jv(-1 / 4, arg_squared)
    J_1_4_arg = sp.jv(1 / 4, arg_squared)

    f_t = (J_minus_1_4_T + T_final * J_minus_5_4_T) * J_1_4_arg \
          - (J_1_4_T - T_final * J_5_4_T) * J_minus_1_4_arg
    return f_t


# 计算f(0)
def compute_f0(T_final=T_final):
    T_half = 0.5 * T_final
    J_minus_1_4 = sp.jv(-1 / 4, T_half)
    J_1_4 = sp.jv(1 / 4, T_half)
    J_minus_5_4 = sp.jv(-5 / 4, T_half)
    J_5_4 = sp.jv(5 / 4, T_half)
    f0 = T_final * J_minus_5_4 * J_1_4 + T_final * J_5_4 * J_minus_1_4
    return f0


# 计算δ_1(t)
def compute_delta1(t, T_final=T_final, f0=None, n_points=10000):
    if f0 is None:
        f0 = compute_f0(T_final)

    tau_grid = np.linspace(0, t, n_points)
    integrand = f0 ** 2 / ((tau_grid + T_final) * compute_f(tau_grid, T_final) ** 2)
    delta = T_final * integrate.trapezoid(integrand, tau_grid)
    return delta


# 计算D_1(t)
def compute_D1(t, T_final=T_final, a=a, f0=None):
    if f0 is None:
        f0 = compute_f0(T_final)

    f_t = compute_f(t, T_final)
    delta_t = compute_delta1(t, T_final, f0)
    denominator = np.sqrt((1 + t / T_final) * (1 + delta_t ** 2))
    D1 = a * f0 / f_t / denominator
    return D1, delta_t, f_t


# 计算A_1(t)的实部和虚部
def compute_A1(t, T_final=T_final, a=a, f0=None):
    D1, delta_t, f_t = compute_D1(t, T_final, a, f0)
    Re_A1 = 0.5 * D1 ** 2

    h = 1e-8
    f_t_plus = compute_f(t + h, T_final)
    f_t_minus = compute_f(t - h, T_final)
    f_dot = (f_t_plus - f_t_minus) / (2 * h)

    Im_A1 = -0.5 * D1 ** 2 * delta_t - 0.5 * f_dot / f_t - 0.25 / (T_final + t)
    return Re_A1, Im_A1, D1, delta_t


# 计算精确波函数
def exact_wavefunction(x, t, n=0, T_final=T_final, a=a):
    f0 = compute_f0(T_final)
    D1, delta_t, f_t = compute_D1(t, T_final, a, f0)
    Re_A1, Im_A1, D1_calc, delta_t_calc = compute_A1(t, T_final, a, f0)

    C = np.sqrt(D1_calc) / (np.pi ** (1 / 4) * np.sqrt(2 ** n * sp.factorial(n)))
    hermite_arg = D1_calc * x
    Hn = sp.hermite(n)(hermite_arg)
    phase = -1j * (n + 0.5) * np.arctan(delta_t_calc)
    A1 = Re_A1 + 1j * Im_A1
    exp_term = np.exp(-A1 * x ** 2)
    psi = C * Hn * np.exp(phase) * exp_term
    return psi


# 计算精确解
psi_exact = np.zeros((N, M), dtype=complex)
n_quantum = 0
f0 = compute_f0(T_final)

for j, t in enumerate(time_grid):
    if j % 100 == 0 and j > 0:
        print(f"精确求解: 时间步 {j}/{N}, t={t:.3f}")
    psi_exact[j, :] = exact_wavefunction(x, t, n=n_quantum, T_final=T_final, a=a)

# 计算精确解模方
norm_exact = np.zeros(N)
for j in range(N):
    norm_exact[j] = np.sum(np.abs(psi_exact[j]) ** 2) * dx

print(f"精确求解完成, 最终模方: {norm_exact[-1]:.8f}")

# 3.二阶分裂方法
print("\n开始二阶分裂方法求解")
# 二阶分裂方法的核心函数
def strang_splitting_step(psi, t, dt, H_kinetic, H_potential):
    """
    二阶Strang分裂方法: e^{dtH/2} ≈ e^{dtT/2} e^{dtV(t+dt/2)} e^{dtT/2}
    """
    # 第一步: 应用 e^{dtT/2}
    psi_half1 = apply_kinetic_evolution(psi, dt / 2, H_kinetic)

    # 第二步: 应用 e^{dtV(t+dt/2)}
    t_mid = t + dt / 2
    V_mid = H_potential(t_mid)
    psi_mid = apply_potential_evolution(psi_half1, dt, V_mid)

    # 第三步: 应用 e^{dtT/2}
    psi_next = apply_kinetic_evolution(psi_mid, dt / 2, H_kinetic)

    return psi_next

def apply_kinetic_evolution(psi, dt, H_kinetic):
    """应用动能演化: e^{-idtH_kinetic} ψ"""
    # 通过傅里叶谱方法实现
    psi_k = np.fft.fft(psi)
    k = 2 * np.pi * np.fft.fftfreq(len(psi), dx)
    exp_factor = np.exp(-1j * dt * 0.5 * k ** 2)
    psi_evolved = np.fft.ifft(exp_factor * psi_k)
    return psi_evolved

def apply_potential_evolution(psi, dt, V):
    """应用势能演化: e^{-idtV} ψ"""
    return np.exp(-1j * dt * V) * psi

# 初始化二阶分裂方法的波函数
psi_strang = psi.copy().astype(complex)
psi_strang_history = np.zeros((N, M), dtype=complex)
psi_strang_history[0] = psi_strang
# 动能算符
def build_kinetic_operator():
    return None  # 在apply_kinetic_evolution中直接实现

# 势能算符
def build_potential_operator(t):
    omega_t = omega(t)
    return 0.5 * omega_t ** 2 * x ** 2

# 二阶分裂方法演化
for j in range(N - 1):
    t_current = j * dt

    # 获取算符
    H_kinetic = build_kinetic_operator()
    H_potential = build_potential_operator

    # 应用二阶分裂
    psi_strang = strang_splitting_step(
        psi_strang, t_current, dt, H_kinetic, H_potential
    )

    # 归一化
    norm = np.linalg.norm(psi_strang) * np.sqrt(dx)
    if norm > 0:
        psi_strang = psi_strang / norm

    psi_strang_history[j + 1] = psi_strang

    if (j + 1) % 100 == 0:
        norm_current = np.sum(np.abs(psi_strang) ** 2) * dx
        print(f"二阶分裂: 时间步 {j + 1}/{N - 1}, 模方: {norm_current:.6f}")

# 计算二阶分裂方法模方
norm_strang = np.zeros(N)
for j in range(N):
    norm_strang[j] = np.sum(np.abs(psi_strang_history[j]) ** 2) * dx

print(f"二阶分裂方法完成, 最终模方: {norm_strang[-1]:.8f}")

# 4. 结果可视化与比较
print("\n开始结果可视化与比较")

# L2 范数误差
error_cn = np.sqrt(np.sum(np.abs(psi_history - psi_exact)**2, axis=1) * dx)
error_strang = np.sqrt(np.sum(np.abs(psi_strang_history - psi_exact)**2, axis=1) * dx)


# 调试信息：检查误差值
print(f"CN误差统计: 最小值={np.min(error_cn):.2e}, 最大值={np.max(error_cn):.2e}, 平均值={np.mean(error_cn):.2e}")
print(f"Strang误差统计: 最小值={np.min(error_strang):.2e}, 最大值={np.max(error_strang):.2e}, 平均值={np.mean(error_strang):.2e}")


# 绘制误差随时间变化的折线图
plt.figure(figsize=(10, 8))

# 绘制两条误差曲线
plt.plot(time_grid, error_cn, 'b-', linewidth=2, label='Crank-Nicolson Error')
plt.plot(time_grid, error_strang, 'r-', linewidth=2, label='Strang Splitting Error')

plt.xlabel('Time t', fontsize=12)
plt.ylabel('Max Absolute Error', fontsize=12)
plt.title('Error Evolution Compared to Exact Solution', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 对数坐标更清晰地显示误差变化

# 添加网格线和y轴范围设置
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.ylim(1e-16, 1)  # 设置合理的y轴范围

plt.tight_layout()
plt.show()

# 额外的诊断图表
plt.figure(figsize=(12, 5))

# 子图1：最终时刻的波函数比较
plt.subplot(1, 2, 1)
plt.plot(x, np.abs(psi_history[-1]), 'b-', linewidth=2, label='CN Numerical', alpha=0.7)
plt.plot(x, np.abs(psi_exact[-1]), 'r--', linewidth=2, label='Exact', alpha=0.7)
plt.plot(x, np.abs(psi_strang_history[-1]), 'g-.', linewidth=2, label='Strang Splitting', alpha=0.7)
plt.xlabel('x')
plt.ylabel('|ψ(x)|')
plt.title('Final Wave Function (t=T)')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：误差的时间演化（线性坐标）
plt.subplot(1, 2, 2)
plt.plot(time_grid, error_cn, 'b-', linewidth=2, label='CN Error')
plt.plot(time_grid, error_strang, 'r-', linewidth=2, label='Strang Error')
plt.xlabel('Time t')
plt.ylabel('Max Absolute Error')
plt.title('Error Evolution (Linear Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 输出最终误差统计
print(f"\n=== 最终误差统计 ===")
print(f"Crank-Nicolson 最终误差: {error_cn[-1]:.2e}")
print(f"Strang Splitting 最终误差: {error_strang[-1]:.2e}")