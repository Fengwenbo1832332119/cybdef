# Offline Intervention (No-Model) — Extended

## 1) Type Sensitivity
- 输出：type_sensitivity.csv / type_sensitivity_MSE.png / type_sensitivity_MCE.png

## 2) Cumulative Deletion Curve
- 输出：cumulative_deletion.csv / cumu_del_curve_MSE.png / cumu_del_curve_MCE.png

## 3) Cross-type Replacement & Unrelated Injection
- 输出：cross_type_and_noise.csv / cross_type_MSE.png / cross_type_MCE.png

**解读建议：**
- 删除特定类型的Δ越大，说明该类型对解释越关键。
- 贪心曲线若显著高于随机曲线，表示系统主要依赖少数“高贡献”证据。
- 注入无关证据(Sleep)若Δ≈0，说明对噪声鲁棒；若Δ>0，说明被噪声误导（应调低该类权重/衰减）。
