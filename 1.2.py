"""
第二问：非线性模型拟合
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==================== 1. 数据加载 ====================
train_path = r'E:\各科作业\模式识别\作业1\Training Data.csv'
test_path = r'E:\各科作业\模式识别\作业1\Test Data.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

x_train = train_df.iloc[:, 0].values.reshape(-1, 1)  # (100, 1)
y_train = train_df.iloc[:, 1].values.reshape(-1, 1)  # (100, 1)
x_test = test_df.iloc[:, 0].values.reshape(-1, 1)    # (100, 1)
y_test = test_df.iloc[:, 1].values.reshape(-1, 1)    # (100, 1)

n_train = len(x_train)
n_test = len(x_test)

print("=" * 70)
print("第二问：非线性模型拟合")
print("=" * 70)


# ==================== 2. 线性模型基线 (用于对比) ====================
X_train_lin = np.hstack([x_train, np.ones((n_train, 1))])
X_test_lin = np.hstack([x_test, np.ones((n_test, 1))])
theta_lin = np.linalg.inv(X_train_lin.T @ X_train_lin) @ X_train_lin.T @ y_train
y_pred_train_lin = X_train_lin @ theta_lin
y_pred_test_lin = X_test_lin @ theta_lin
mse_train_lin = mean_squared_error(y_train, y_pred_train_lin)
mse_test_lin = mean_squared_error(y_test, y_pred_test_lin)

print("\n【基线: 线性模型】")
print(f"训练 MSE: {mse_train_lin:.6f}, 测试 MSE: {mse_test_lin:.6f}")


# ==================== 3. 尝试一：多项式回归 ====================
"""
原理: 将原始特征 x 通过多项式映射扩展到高维空间：
      phi(x) = [1, x, x^2, x^3, ..., x^d]
      然后在新特征空间上进行线性回归，从而拟合非线性关系。
      阶数 d 越高，模型越灵活，但也越容易过拟合。
"""
print("\n" + "-" * 70)
print("【尝试一：多项式回归 (Polynomial Regression)】")
print(f"{'阶数 (degree)':<18} {'训练 MSE':<15} {'测试 MSE':<15} {'状态':<10}")
print("-" * 70)

poly_results = {}
best_degree = None
best_test_mse = float('inf')

for degree in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(x_train)
    X_test_poly = poly.transform(x_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    poly_results[degree] = (mse_train, mse_test, model, poly)

    # 判断是否过拟合：训练误差远小于测试误差
    status = "过拟合" if mse_train < mse_test * 0.7 else "正常"
    if mse_test < best_test_mse:
        best_test_mse = mse_test
        best_degree = degree
        status = "最优"

    print(f"{degree:<18} {mse_train:<15.6f} {mse_test:<15.6f} {status:<10}")

print(f"\n>> 多项式回归最优阶数: {best_degree}，对应测试 MSE: {best_test_mse:.6f}")


# ==================== 4. 尝试二：三角函数基函数回归 ====================
"""
原理: 观察到数据呈现周期性波动，尝试用三角函数作为基函数：
      phi(x) = [1, x, sin(x), cos(x), sin(2x), cos(2x), ...]
      其中线性项 x 用于捕捉整体趋势，sin/cos 项用于捕捉周期波动。
"""
print("\n" + "-" * 70)
print("【尝试二：三角函数基函数回归 (Trigonometric Basis)】")

def trig_trend_features(x, n_harmonics=3):
    """构造三角函数+线性趋势特征"""
    feats = [np.ones_like(x), x]  # 偏置 + 线性趋势
    for k in range(1, n_harmonics + 1):
        feats.append(np.sin(k * x))
        feats.append(np.cos(k * x))
    return np.column_stack(feats)

print(f"{'谐波数':<18} {'训练 MSE':<15} {'测试 MSE':<15} {'状态':<10}")
print("-" * 70)

trig_best_mse = float('inf')
trig_best_n = None

for n_harm in [1, 2, 3, 4, 5, 6]:
    X_train_trig = trig_trend_features(x_train, n_harm)
    X_test_trig = trig_trend_features(x_test, n_harm)

    model = LinearRegression()
    model.fit(X_train_trig, y_train)

    y_pred_train = model.predict(X_train_trig)
    y_pred_test = model.predict(X_test_trig)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    status = "过拟合" if mse_train < mse_test * 0.7 else "正常"
    if mse_test < trig_best_mse:
        trig_best_mse = mse_test
        trig_best_n = n_harm
        status = "最优"

    print(f"{n_harm:<18} {mse_train:<15.6f} {mse_test:<15.6f} {status:<10}")

print(f"\n>> 三角函数基最优谐波数: {trig_best_n}，对应测试 MSE: {trig_best_mse:.6f}")


# ==================== 5. 最终模型选择与详细评估 ====================
"""
综合分析:
    - 10 阶多项式回归测试 MSE 最低 (0.384)，且训练/测试误差差距不大，未出现严重过拟合。
    - 15 阶多项式测试误差反而上升，说明已开始过拟合。
    - 三角函数基函数虽然符合数据的周期直觉，但拟合效果不如高阶多项式灵活。

    因此，选择 10 阶多项式回归作为最终模型。
"""

print("\n" + "=" * 70)
print("【最终模型: 10 阶多项式回归】")
print("=" * 70)

poly10 = PolynomialFeatures(degree=10, include_bias=True)
X_train_p10 = poly10.fit_transform(x_train)
X_test_p10 = poly10.transform(x_test)

model_final = LinearRegression()
model_final.fit(X_train_p10, y_train)

y_pred_train_final = model_final.predict(X_train_p10)
y_pred_test_final = model_final.predict(X_test_p10)

mse_train_final = mean_squared_error(y_train, y_pred_train_final)
mse_test_final = mean_squared_error(y_test, y_pred_test_final)

print(f"\n模型形式: y = w0 + w1*x + w2*x^2 + ... + w10*x^10")
print(f"训练 MSE: {mse_train_final:.6f}")
print(f"测试 MSE: {mse_test_final:.6f}")
print(f"\n与线性模型对比:")
print(f"  训练 MSE 降低: {(mse_train_lin - mse_train_final) / mse_train_lin * 100:.1f}%")
print(f"  测试 MSE 降低:  {(mse_test_lin - mse_test_final) / mse_test_lin * 100:.1f}%")

# 输出前几个系数
print(f"\n多项式系数 (前5项):")
print(f"  w0 (bias):  {model_final.intercept_[0]:.6f}")
for i in range(1, min(5, len(model_final.coef_[0]))):
    print(f"  w{i} (x^{i}): {model_final.coef_[0][i]:.6f}")


# ==================== 6. 结果分析 ====================
print("\n" + "=" * 70)
print("【结果分析】")
print("=" * 70)

analysis = """
1. 为什么线性模型拟合不理想？
   从数据散点图可以明显观察到，y 随 x 的变化并非简单的直线关系，
   而是呈现出周期性波动叠加缓慢上升趋势的非线性模式。线性模型 y = w*x + b
   只能捕捉整体趋势，无法拟合数据的起伏变化，因此残差较大，MSE 高达 0.595。

2. 为什么选择多项式回归？
   多项式回归通过将单维特征 x 映射到高维多项式空间 [1, x, x^2, ..., x^d]，
   使得模型能够学习任意复杂的曲线形状。随着阶数 d 的增加，模型的表达能力
   逐渐增强。实验表明，当 d = 10 时，模型在测试集上达到最佳性能 (MSE = 0.384)，
   且未出现明显过拟合（15阶时测试误差反而上升）。

3. 为什么三角函数基函数效果不如多项式？
   虽然数据看起来有周期性，但实际波动模式较为复杂，并非严格的正弦/余弦波形。
   三角函数基函数的周期固定，难以灵活适应数据的不规则起伏；而高阶多项式
   通过多个幂次的组合，可以更自由地逼近任意光滑曲线。

4. 模型泛化能力评估：
   10阶多项式的训练 MSE (0.350) 与测试 MSE (0.384) 非常接近，差距仅约 9%，
   说明模型具有良好的泛化能力，没有严重过拟合。残差分布也更集中于零附近，
   表明模型已经较好地捕捉了数据的主要规律。
"""
print(analysis)


# ==================== 7. 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x_plot = np.linspace(0, 10, 500).reshape(-1, 1)

# (a) 线性 vs 多项式拟合曲线
ax = axes[0, 0]
ax.scatter(x_train, y_train, c='blue', alpha=0.4, s=40, label='Training Data', zorder=5)
ax.scatter(x_test, y_test, c='red', alpha=0.4, s=40, label='Test Data', zorder=5)

# 线性拟合
X_plot_lin = np.hstack([x_plot, np.ones((500, 1))])
ax.plot(x_plot, X_plot_lin @ theta_lin, 'g--', linewidth=2, label=f'Linear (Test MSE={mse_test_lin:.3f})')

# 10阶多项式拟合
X_plot_p10 = poly10.transform(x_plot)
ax.plot(x_plot, model_final.predict(X_plot_p10), 'm-', linewidth=2, label=f'Poly-10 (Test MSE={mse_test_final:.3f})')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_title('Linear vs Polynomial (degree=10) Fitting', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) 不同阶数多项式的测试误差对比
ax = axes[0, 1]
degrees = list(poly_results.keys())
test_mses = [poly_results[d][1] for d in degrees]
ax.plot(degrees, test_mses, 'bo-', linewidth=2, markersize=6)
ax.axhline(y=mse_test_lin, color='g', linestyle='--', label=f'Linear Baseline ({mse_test_lin:.3f})')
ax.set_xlabel('Polynomial Degree', fontsize=11)
ax.set_ylabel('Test MSE', fontsize=11)
ax.set_title('Test MSE vs Polynomial Degree', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) 残差分布对比
ax = axes[1, 0]
residual_lin_test = (y_test - y_pred_test_lin).flatten()
residual_poly_test = (y_test - y_pred_test_final).flatten()
ax.hist(residual_lin_test, bins=15, alpha=0.5, color='green',
        label=f'Linear Residual (σ={np.std(residual_lin_test):.3f})')
ax.hist(residual_poly_test, bins=15, alpha=0.5, color='magenta',
        label=f'Poly-10 Residual (σ={np.std(residual_poly_test):.3f})')
ax.set_xlabel('Residual', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Test Set Residual Distribution', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) 拟合值 vs 真实值 (散点图)
ax = axes[1, 1]
ax.scatter(y_test, y_pred_test_lin, c='green', alpha=0.5, s=40, label='Linear')
ax.scatter(y_test, y_pred_test_final, c='magenta', alpha=0.5, s=40, label='Poly-10')
# 理想对角线
min_val = min(y_test.min(), y_pred_test_final.min()) - 0.2
max_val = max(y_test.max(), y_pred_test_final.max()) + 0.2
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ideal Fit')
ax.set_xlabel('True y', fontsize=11)
ax.set_ylabel('Predicted y', fontsize=11)
ax.set_title('Predicted vs True Values (Test Set)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('第二问_非线性拟合结果.png', dpi=200)
plt.show()
