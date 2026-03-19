import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': ['SimHei', 'DejaVu Sans'],  # 中文字体优先，备用英文字体
    'axes.unicode_minus': False,  # 正确显示负号
    'mathtext.fontset': 'stix',  # 数学字体
    'figure.autolayout': True,  # 自动调整布局
})


def shooting_method_bvp(f, a, b, alpha, beta, gamma_guess, h=0.1, tol=1e-6, max_iter=100):
    """
    打靶法求解二阶边值问题: y'' = f(x, y, y'), y(a)=alpha, y(b)=beta

    参数:
        f: 函数，表示微分方程 f(x, y, y')
        a, b: 区间端点
        alpha, beta: 边界条件
        gamma_guess: 初始斜率猜测值
        h: 步长
        tol: 误差阈值
        max_iter: 最大迭代次数

    返回:
        x_vals: x值数组
        y_vals: y值数组
        final_gamma: 最终找到的初始斜率
    """

    def rk4_step(f, x, y, v, h):
        """四阶Runge-Kutta法单步（用于一阶方程组）"""
        k1_y = v
        k1_v = f(x, y, v)

        k2_y = v + 0.5 * h * k1_v
        k2_v = f(x + 0.5 * h, y + 0.5 * h * k1_y, v + 0.5 * h * k1_v)

        k3_y = v + 0.5 * h * k2_v
        k3_v = f(x + 0.5 * h, y + 0.5 * h * k2_y, v + 0.5 * h * k2_v)

        k4_y = v + h * k3_v
        k4_v = f(x + h, y + h * k3_y, v + h * k3_v)

        y_new = y + (h / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        v_new = v + (h / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return y_new, v_new

    def solve_ivp(gamma):
        """求解给定初始斜率的初值问题"""
        x = a
        y = alpha
        v = gamma

        x_list = [x]
        y_list = [y]

        while x < b - 1e-10:
            y, v = rk4_step(f, x, y, v, h)
            x += h
            x_list.append(x)
            y_list.append(y)

        return np.array(x_list), np.array(y_list), y_list[-1]

    def error_function(gamma):
        """误差函数: 在x=b处的值与目标值的差"""
        _, _, y_b = solve_ivp(gamma)
        return y_b - beta

    # 找到两个gamma值使误差异号
    gamma1 = gamma_guess
    error1 = error_function(gamma1)

    # 寻找另一个gamma值
    if error1 > 0:
        gamma2 = gamma1 - 0.1
    else:
        gamma2 = gamma1 + 0.1

    error2 = error_function(gamma2)

    # 确保两个初始猜测值误差异号
    while error1 * error2 > 0 and max_iter > 0:
        # 根据误差符号和大小综合判断
        if error1 > 0:
            # 两个误差都大于 0，需要减小 gamma
            if abs(error2) > abs(error1):
                gamma2 = gamma1 - abs(gamma2 - gamma1) * 2
            else:
                gamma1 = gamma2 - abs(gamma1 - gamma2) * 2
        else:
            # 两个误差都小于 0，需要增大 gamma
            if abs(error2) > abs(error1):
                gamma2 = gamma1 + abs(gamma2 - gamma1) * 2
            else:
                gamma1 = gamma2 + abs(gamma1 - gamma2) * 2

        error1 = error_function(gamma1)
        error2 = error_function(gamma2)
        max_iter -= 1

    # 使用scipy的求根函数
    try:
        #二分法
        result = root_scalar(error_function, bracket=[min(gamma1, gamma2), max(gamma1, gamma2)],
                             method='bisect', xtol=tol)
        final_gamma = result.root
    except:
        # 割线法
        result = root_scalar(error_function, x0=gamma_guess, fprime=None, method='secant')
        final_gamma = result.root

    # 使用最终找到的gamma求解
    x_vals, y_vals, _ = solve_ivp(final_gamma)

    return x_vals, y_vals, final_gamma


def shooting_method_eigenvalue(p, a, b, lambda_guess, h=0.1, tol=1e-6):
    """
    打靶法求解特征值问题: y'' + λ * p(x) * y = 0, y(a)=0, y(b)=0

    参数:
        p: 函数，p(x)
        a, b: 区间端点
        lambda_guess: 特征值初始猜测
        h: 步长
        tol: 误差阈值

    返回:
        eigenvalue: 找到的特征值
        x_vals: x值数组
        y_vals: 特征函数（归一化）
    """

    def f_eigen(x, y, v, lam):
        """特征值问题的微分方程: y'' = -λ * p(x) * y"""
        return -lam * p(x) * y

    def rk4_step_eigen(f, x, y, v, lam, h):
        """四阶Runge-Kutta法单步"""
        k1_y = v
        k1_v = f(x, y, v, lam)

        k2_y = v + 0.5 * h * k1_v
        k2_v = f(x + 0.5 * h, y + 0.5 * h * k1_y, v + 0.5 * h * k1_v, lam)

        k3_y = v + 0.5 * h * k2_v
        k3_v = f(x + 0.5 * h, y + 0.5 * h * k2_y, v + 0.5 * h * k2_v, lam)

        k4_y = v + h * k3_v
        k4_v = f(x + h, y + h * k3_y, v + h * k3_v, lam)

        y_new = y + (h / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        v_new = v + (h / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return y_new, v_new

    def solve_eigen_ivp(lam):
        """求解给定λ的初值问题: y(a)=0, y'(a)=1"""
        x = a
        y = 0.0
        v = 1.0  # 固定初始斜率，文档10.8节提到由于齐次性可任意缩放

        x_list = [x]
        y_list = [y]

        while x < b - 1e-10:
            y, v = rk4_step_eigen(f_eigen, x, y, v, lam, h)
            x += h
            x_list.append(x)
            y_list.append(y)

        return np.array(x_list), np.array(y_list), y_list[-1]

    def error_function_eigen(lam):
        """误差函数: 在x=b处的值"""
        _, _, y_b = solve_eigen_ivp(lam)
        return y_b  # 目标是y(b)=0

    # 使用求根算法寻找特征值
    result = root_scalar(error_function_eigen, x0=lambda_guess, fprime=None,
                         method='secant', xtol=tol)
    eigenvalue = result.root

    # 使用找到的特征值求解特征函数
    x_vals, y_vals, _ = solve_eigen_ivp(eigenvalue)

    # 归一化特征函数（使其最大绝对值为1）
    y_vals = y_vals / np.max(np.abs(y_vals))

    return eigenvalue, x_vals, y_vals


# ===================== 示例和测试 =====================

if __name__ == "__main__":
    print("打靶法求解边值问题示例")

    # 示例1:y'' - y = 0, y(0)=0, y(1)=1
    print("\n1. 文档示例: y'' - y = 0, y(0)=0, y(1)=1")
    print("   解析解: y(x) = (e^x - e^{-x})/(e - e^{-1})")
    print("   精确初始斜率: y'(0) ≈ 0.850918")


    def f_example1(x, y, v):
        return y  # y'' = y


    # 使用打靶法求解
    x_vals, y_vals, gamma = shooting_method_bvp(f_example1, 0, 1, 0, 1,
                                                gamma_guess=0.5, h=0.05)

    print(f"   数值解初始斜率: {gamma:.6f}")
    print(f"   在x=1处的值: {y_vals[-1]:.6f}")
    print(f"   绝对误差: {abs(y_vals[-1] - 1):.2e}")

    # 计算解析解用于比较
    analytical = (np.exp(x_vals) - np.exp(-x_vals)) / (np.exp(1) - np.exp(-1))
    max_error = np.max(np.abs(y_vals - analytical))
    print(f"   最大误差: {max_error:.2e}")

    print("\n" + "=" * 60)
    print("打靶法求解特征值问题示例")

    # 示例2: y'' + λ^2 y = 0, y(0)=y(π)=0
    print("\n2. 特征值问题: y'' + λ^2 y = 0, y(0)=y(π)=0")
    print("   精确特征值: λ = 1, 2, 3, ...")


    def p_func(x):
        return 1.0  # 对于这个例子，p(x)=1


    # 使用打靶法寻找最小特征值
    eigenvalue, x_eigen, y_eigen = shooting_method_eigenvalue(p_func, 0, np.pi,
                                                              lambda_guess=0.5, h=0.05)

    print(f"   数值最小特征值: {eigenvalue:.6f}")
    print(f"   精确最小特征值: 1.0")
    print(f"   相对误差: {abs(eigenvalue - 1.0):.2e}")

    # 验证归一化结果
    print(f"   归一化后特征函数最大值: {np.max(np.abs(y_eigen)):.6f}")

    print("\n" + "=" * 60)
    print("附加示例: 更复杂的边值问题")

    # 示例3: 非线性边值问题 y'' = -sin(y), y(0)=0, y(π/2)=1
    print("\n4. 非线性问题: y'' = -sin(y), y(0)=0, y(π/2)=1")


    def f_nonlinear(x, y, v):
        return -np.sin(y)


    x_vals_nl, y_vals_nl, gamma_nl = shooting_method_bvp(f_nonlinear, 0, np.pi / 2,
                                                         0, 1, gamma_guess=1.0, h=0.01)

    print(f"   找到的初始斜率: {gamma_nl:.6f}")
    print(f"   在x=π/2处的值: {y_vals_nl[-1]:.6f}")

    # 可视化结果

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 图1: 边值问题解
    ax1 = axes[0, 0]
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='数值解')
    ax1.plot(x_vals, analytical, 'r--', linewidth=2, label='解析解', alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y(x)')
    ax1.set_title('边值问题: y\'\' - y = 0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 误差
    ax2 = axes[0, 1]
    error = y_vals - analytical
    ax2.plot(x_vals, error, 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('误差')
    ax2.set_title('数值解与解析解的误差')
    ax2.grid(True, alpha=0.3)

    # 图3: 特征函数
    ax3 = axes[1, 0]
    ax3.plot(x_eigen, y_eigen, 'm-', linewidth=2)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y(x)')
    ax3.set_title(f'特征函数 (λ={eigenvalue:.4f})')
    ax3.grid(True, alpha=0.3)

    # 添加归一化验证标注
    ax3.text(0.5, 0.9, f'归一化: max(|y|)={np.max(np.abs(y_eigen)):.2f}',
             transform=ax3.transAxes, ha='center', fontsize=10)

    # 图4: 非线性问题
    ax4 = axes[1, 1]
    ax4.plot(x_vals_nl, y_vals_nl, 'c-', linewidth=2)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y(x)')
    ax4.set_title('非线性问题: y\'\' = -sin(y)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n代码执行完成！")
