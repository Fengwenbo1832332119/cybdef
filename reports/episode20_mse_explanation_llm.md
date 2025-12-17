# 可解释强化学习解释性分析报告：Episode 20

本分析针对 Episode 20 中蓝方特工的决策过程进行深入解释，重点关注策略变迁、规则驱动因素以及解释质量。

---

## 1. 最小证据集（Minimal Supporting Evidence, MSE）概念与作用

### 1.1 最小证据集定义

最小证据集（MSE）是指支持强化学习智能体做出特定决策或遵循特定策略所需的最小、非冗余的环境事实或观察集合。在可解释强化学习（XRL）中，MSE 的目标是剥离所有不必要的背景信息，仅留下那些在因果上驱动决策的关键状态变量。

### 1.2 在 Episode 20 中的作用

Episode 20 经历了从“平静”（Quiet, Step 1-2）到“可疑活动”（Suspicious Activity, Step 3-7）再到短暂平静（Step 8）的急剧状态转换。

MSE 在本 Episode 中的作用是**精准定位策略转换的触发器**。

在 Step 1-2，蓝方的规则和推荐行动（如 `MonitorTraffic`）由 `NOT FACT('suspicious_activity')` 驱动。然而，在 Step 3，一旦检测到 `FACT('suspicious_activity')`，MSE 立刻识别出此事实是激活所有防御和反击策略（如高优先级的 `InvestigateHost` 和 `Decoy` 部署）的非协商前提。通过 MSE，我们可以确定蓝方策略的激进化是完全基于环境事实的根本变化。

---

## 2. 代表性策略的证据分析

由于 Episode 20 在 Step 3 出现了关键的状态转变，我们选取一个平静状态的动作、一个高优先级推荐动作和一个实际执行的防御动作进行分析。

| 步骤 | 动作类型 | 具体动作 (Action) | 关键驱动规则 (Driving Rules) |
| :--- | :--- | :--- | :--- |
| **Step 3 (推荐)** | 侦查 (Investigate) | `InvestigateHost_User4` | Prefer-Investigate-On-Suspicion; Investigate-Priority-Over-Decoy |
| **Step 3 (执行)** | 诱捕 (Decoy) | `DecoySSHD_User2` | Prefer-Decoy-Under-Suspicion |

### 2.1 策略分析一：高优先级侦查 (`InvestigateHost_User4`)

**动作：** `InvestigateHost_User4` (Step 3 推荐动作，优先级 1.1)

**对应的 MSE 证据：**
1. `FACT('suspicious_activity')`：环境中检测到可疑活动。
2. `NOT FACT('host_compromised')`：尚未确认主机已被完全攻陷。

**解释与合理性：**
该动作在 Step 3 获得极高的推荐优先级 (1.1)，主要由两条规则驱动：
*   'Prefer-Investigate-On-Suspicion' (P +0.7)
*   'Investigate-Priority-Over-Decoy' (P +0.4)
这两条规则的共同条件是 `FACT('suspicious_activity')` 存在，但尚未达到最危险的 `FACT('host_compromised')` 状态。策略目标是在攻击者成功建立持久控制前，通过快速侦查来定位威胁。

**如果缺少关键证据 $FACT('suspicious\_activity')$：**
如果缺少此证据，环境将处于“平静”状态。此时，上述两条高优先级规则将不会被激活。取而代之的是“平静时避免删除诱饵/恶意软件”（Avoid-Remove-Decoy-When-Quiet）和“平静时优先监控”（Prefer-Monitor-When-Quiet）等规则。在这种情况下，`InvestigateHost` 的优先级将远低于 1.1，高优先级的侦查决策将变得**不合理**。因此，`FACT('suspicious_activity')` 是决定此决策合理性的核心证据。

### 2.2 策略分析二：部署诱捕服务 (`DecoySSHD_User2`)

**动作：** `DecoySSHD_User2` (Step 3 蓝方实际执行动作)

**对应的 MSE 证据：**
1. `FACT('suspicious_activity')`：环境中检测到可疑活动。

**解释与合理性：**
该动作由规则 'Prefer-Decoy-Under-Suspicion' (P +0.22) 激活。虽然侦查动作拥有更高的总优先级 (1.1)，但蓝方特工选择在 Step 3 执行诱捕操作，表明其决策模型在检测到可疑活动后，启动了**分散注意力的策略**，试图干扰攻击者的侦查和横向移动。

**如果缺少关键证据 $FACT('suspicious\_activity')$：**
如果缺少此证据，即环境处于平静状态，则规则 'Avoid-Remove-Decoy-When-Quiet' 将被激活。该规则会对所有 Decoy 动作应用 **`hard_mask`**（硬屏蔽），彻底禁止该动作的执行。因此，部署诱捕服务的决策将是**不合理且被策略禁止**的。`FACT('suspicious_activity')` 是执行任何防御性诱捕动作的**必要条件**。

---

## 3. 解释质量评估与结论

对本 Episode 的 PolicySpeak 生成评测指标进行分析，以评估解释的**忠诚度**（Faithfulness）和**稳定性**（Stability）。

### 3.1 忠诚度评估 (Faithfulness)

| 指标 | 结果 | 解释 |
| :--- | :--- | :--- |
| **Hallucination@0** | $0.0000$ | 完美忠诚度。此指标衡量解释中是否缺少了驱动决策的关键事实。结果为 0.0000 表明所有在规则推理中必要的证据都被成功地提取并包含在解释中。蓝方决策过程的解释是**完全忠于证据**的。 |

### 3.2 稳定性与可靠性评估

| 指标 | 结果 | 解释 |
| :--- | :--- | :--- |
| **同证据一致性 (Consistency)** | $1.0000$ | 极高稳定性。此指标表示解释机制在多次运行或不同情境下，能否持续将决策归因于相同的最小证据集。结果为 1.0000 表明智能体的决策逻辑在 Episode 20 中是**高度稳定且可重复验证**的。 |
| **Calibration-E (Brier/NLL)** | $nan$ | 由于数据限制，校准指标未计算，无法直接评估模型置信度与实际决策结果的吻合程度。 |

### 3.3 总结

Episode 20 的解释性分析显示出极高的质量。蓝方在 Step 1-7 的决策过程（从预防性监控到紧急侦查/诱捕）被明确地归因于环境状态 `FACT('suspicious_activity')` 的转变。

*   **结论：** 基于 $Hallucination@0 = 0.0000$ 和 $Consistency = 1.0000$，本 Episode 的可解释强化学习模型提供了**忠诚、稳定且可信赖**的解释，成功地将复杂的策略转换（由规则驱动的优先级变化）溯源至最小的关键环境事实集。这证明了基于规则的 XRL 框架在网络安全场景中捕捉决策因果关系的有效性。