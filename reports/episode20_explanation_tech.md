# Episode 20 对抗对局蓝队策略技术分析报告

## I. 整体态势概述

Episode 20 的对局可分为三个阶段：初始静默侦察阶段（Step 1-2）、可疑活动爆发与防御部署阶段（Step 3-7），以及威胁消退后的清理侦察阶段（Step 8）。

蓝队在 Step 3 首次侦测到 `suspicious_activity`，引发了防御策略的急剧转变。红队（攻击方）的活动并未导致明确的 `host_compromised` 事实被蓝队发现，这使得蓝队的重点从清除转向了调查和欺骗。蓝队的实际行动显示出一种混合策略：在规则强烈推荐进行主机调查（InvestigateHost）时，蓝队首先选择了部署蜜罐（Decoy），以期拖延或误导攻击者。

### 攻击链演进推测

鉴于蓝队在 Step 3 首次检测到可疑活动，我们可以推测攻击者在 Step 1-2 进行了侦察或植入准备，并在 Step 3 尝试执行了初步访问（Initial Access）或本地侦察活动。

| 攻击阶段 | 涉及时间线 | 关键指标 / 蓝队行动 |
| :--- | :--- | :--- |
| **侦察 / 潜伏** | Step 1-2 | 环境处于安静状态，蓝队执行主动侦察 (`InvestigateHost_Enterprise1`, `InvestigateHost_Op_Server0`)。 |
| **初步访问 (Initial Access)** | Step 3 | 触发 `suspicious_activity` 事实。蓝队响应以 Decoy 动作 (`DecoySSHD_User2`) 为主。 |
| **持久化 / 规避** | Step 4, 6 | 蓝队继续部署诱饵服务 (`DecoySmss_Enterprise1`, `DecoyTomcat_Enterprise0`)，试图混淆攻击者。 |
| **异常行为 / 响应** | Step 5 | 蓝队执行了反常的 `RestoreService_User4`，暗示 User4 主机上可能存在服务中断或配置错误。 |
| **调查 / 缓解** | Step 7-8 | 蓝队开始执行调查 (`InvestigateHost_Op_Host0`, `InvestigateHost_User4`)，并在威胁消退后继续保持调查态势。 |

## II. 威胁与防御演进细节

### 阶段一：静默侦察 (Step 1-2)

在没有环境威胁信号时，蓝队规则系统主要由“安静”规则驱动。

1.  **规则触发:** `Avoid-Remove-Decoy-When-Quiet` 屏蔽了移除恶意软件和部署诱饵的动作。`Prefer-Monitor-When-Quiet` 推荐 `MonitorTraffic` (P=0.4)。
2.  **蓝队决策:** 蓝队忽略了推荐的被动监控，转而执行了两次主动调查：`InvestigateHost_Enterprise1` (Step 1) 和 `InvestigateHost_Op_Server0` (Step 2)。这表明蓝队倾向于在平静期进行资产清点和健康检查。

### 阶段二：可疑活动响应 (Step 3-7)

Step 3 标志着环境状态的重大转变，`suspicious_activity` 事实的触发激活了一系列高优先级响应规则。

1.  **Step 3 规则分析:**
    *   `FACT('suspicious_activity')` 触发。
    *   `Prefer-Investigate-On-Suspicion` (P=+0.7) 和 `Investigate-Priority-Over-Decoy` (P=+0.4) 叠加，使得 `InvestigateHost` 动作获得了极高的优先级（P $\ge 1.1$）。
    *   `Prefer-Decoy-Under-Suspicion` 赋予 Decoy 动作优先级 P=+0.22。

2.  **蓝队决策链条（Step 3, 4, 6）：欺骗优先于侦察**
    *   尽管规则强烈推荐调查（P $\ge 1.1$），蓝队在 Step 3、4、6 连续部署了诱饵服务（`DecoySSHD_User2`, `DecoySmss_Enterprise1`, `DecoyTomcat_Enterprise0`）。这是一种偏离规则推荐的防御策略，即在威胁初期优先进行环境混淆和时间争取。

3.  **Step 5 异常动作：服务恢复**
    *   在持续存在可疑活动的情况下，蓝队执行了 `RestoreService_User4`。标准操作中，服务恢复通常在确定服务已受损并完成修复后执行。在未进行 Investigation 并确认威胁源的情况下进行 Restore，是一个风险较高的动作，可能导致攻击者恢复其利用的服务或配置。

4.  **Step 7 转向侦察:**
    *   在部署了三次诱饵和一次服务恢复后，蓝队终于执行了规则高优先级推荐的动作：`InvestigateHost_Op_Host0`。

### 阶段三：威胁消退 (Step 8)

在 Step 8，环境事实中不再包含 `suspicious_activity`，系统恢复到静默状态。

1.  **规则触发:** 静默规则重新激活，推荐 `MonitorTraffic` (P=0.4)。
2.  **蓝队决策:** 蓝队继续执行主动调查 `InvestigateHost_User4`，这与 Step 1-2 的模式一致，即在无威胁时也偏好主动审计而非被动监控。

## III. 蓝方关键决策解析与规则作用

本局的关键在于 Step 3 蓝队对高优先级 `Investigate` 推荐的策略性忽略，转而执行 `Decoy`。

| 决策点 (Step) | 环境事实 (FACT) | 规则推荐 (Priority) | 蓝方实际行动 | 策略解读 |
| :--- | :--- | :--- | :--- | :--- |
| **1-2** | NOT `suspicious_activity` | MonitorTraffic (P=0.4) | InvestigateHost | 偏好主动资产审计。 |
| **3** | `suspicious_activity` | InvestigateHost (P $\ge 1.1$) | DecoySSHD_User2 | 优先度最高的 Investigate 被低优先级的 Decoy (P=0.22) 替代，显示出蓝队对即时欺骗的重视。 |
| **5** | `suspicious_activity` | InvestigateHost (P $\ge 1.1$) | RestoreService_User4 | 异常动作。可能表明在 User4 上检测到服务降级或拒绝服务，并试图恢复，但缺乏威胁根源的调查。 |

### 规则与证据链分析

本轮对局中，CSKG 规则成功地根据环境事实动态调整了动作优先级：

1.  **静默期规则 (Steps 1-2, 8):**
    *   **触发条件:** `NOT FACT('suspicious_activity')` AND `NOT FACT('host_compromised')`
    *   **影响:** 严格限制了高风险动作（如移除和诱饵部署），并引导策略偏向监控（P=0.4）。
2.  **活动期规则 (Steps 3-7):**
    *   **触发条件:** `FACT('suspicious_activity')`
    *   **影响:** 通过叠加优先级（P=0.7 + P=0.4），明确将调查定位为最高优先级的响应动作。
    *   **结果:** 蓝方执行了**反规则推荐**的 Decoy 动作，这通常表示蓝队策略中存在一个隐性的、高权重的偏好，即“先放诱饵、再调查”，这个偏好超越了当前显式规则所赋予的优先级。

## IV. 结构化策略（PolicySpeak）与评测指标分析

### PolicySpeak JSON 与 Timeline 的严重不一致性

对提供的 PolicySpeak JSON 进行分析时，发现其内容与 Episode 20 的实际执行轨迹存在根本性矛盾：

1.  **事实引用错误（Citation Hallucination）：** PolicySpeak JSON 引用了事实 `Suspicious activity has been detected (episode=4, step=4)`。这与我们当前分析的 Episode 20 事实引用不一致。在 Episode 20 中，可疑活动最早在 Step 3 被检测到。
2.  **行动不匹配：** PolicySpeak JSON 推荐的行动是 `DecoyVsftpd_Enterprise0`。然而，蓝队在 Episode 20, Step 4 的实际动作是 `DecoySmss_Enterprise1`。
    *   **结论:** 提供的 PolicySpeak JSON 似乎是一个来自不同对局（Episode 4）或不同时间点的策略输出，而非 Episode 20 蓝队实际执行策略的忠实记录。

### 评测指标的技术点评

| 指标 | 数值 | 技术解读 |
| :--- | :--- | :--- |
| **Hallucination@0** | 0.0000 | 该指标在 PolicySpeak 内部是完美的。它表示在生成摘要或报告时，没有引入与 JSON 结构化证据（`evidence` 字段）不符的虚假信息。**但这并不保证 JSON 证据本身（如“episode=4, step=4”）是准确的。** |
| **同证据一致性 (Consistency)** | 1.0000 | PolicySpeak 的规划 (`plans` 字段) 忠实地反映了策略 (`policies` 字段) 中基于证据所定义的行动。即，如果证据 `step4_fact_suspicious` 触发了策略，则该策略定义的 `DecoyVsftpd_Enterprise0` 动作会出现在计划中，达到了内部一致性。 |
| **Calibration-E (Brier/NLL)** | nan | 校准指标缺失（nan）。无法评估策略对证据置信度的概率性预测能力。 |

**总结：** PolicySpeak JSON 在**内部一致性**方面表现完美 (Consistency = 1.0000; Hallucination@0 = 0.0000)，即其生成的文本和规划完全遵循其内部引用的证据和定义的策略。然而，由于 PolicySpeak 引用了错误的 Episode/Step 编号，且其推荐动作与实际蓝方动作不符，该 PolicySpeak JSON **缺乏外部保真度**，无法作为 Episode 20 蓝队行为的准确解释。

---
*（注：本分析基于提供的 PolicySpeak JSON 结构，即使其与 Timeline 数据存在冲突，也必须对其内部逻辑进行评估。）*