# Episode 20 蓝队行动复盘：告警后的诱捕与调查节奏

## I. 整体态势概述

Episode 20 呈现了一个经典的“静默期突发告警”场景。蓝队在最初的静默期进行了例行性的主机调查，但在 Step 3 突然触发了“可疑活动”告警，将态势瞬间推向高风险状态。蓝队的反应策略表现出明显的偏向：尽管规则引擎强烈推荐执行**调查（InvestigateHost）**，蓝方行动却优先执行了**诱捕/欺骗（Decoy）**操作，以争取时间或扰乱攻击者，随后才开始有限的调查。

最终，在 Step 8，环境事实回归静默状态，表明威胁事件可能已被处理或攻击者被有效干扰并暂停了行动。

## II. 威胁与防御演进（Timeline Analysis）

| 步骤 (Step) | 环境事实 (Facts) | 蓝队推荐动作 (Recommended) | 蓝队实际动作 (Action) | 关键发现与战术意图 |
| :---: | :--- | :--- | :--- | :--- |
| **1-2** | 静默期 (Reward 0.0) | MonitorTraffic (优先级 0.4) | InvestigateHost_Enterprise1, InvestigateHost_Op_Server0 | **主动侦察：** 蓝队未遵循最高优先级的监控建议，而是进行了主动主机调查。 |
| **3 (高优先级)** | **Suspicious activity detected** | InvestigateHost (优先级 1.1) | **DecoySSHD_User2** | **战术冲突：** 告警触发，规则强烈要求调查。蓝队选择执行 Decoy 诱捕，目标是 User2。 |
| **4** | Suspicious activity 持续 | InvestigateHost (优先级 1.1) | **DecoySmss_Enterprise1** | **持续欺骗：** 继续部署诱捕，目标 Enterprise1。 |
| **5** | Suspicious activity 持续 | InvestigateHost (优先级 1.1) | RestoreService_User4 | **异常行为：** 在高警报期执行服务恢复操作，可能用于清理或恢复被中断的服务，但优先级存疑。 |
| **6** | Suspicious activity 持续 | InvestigateHost (优先级 1.1) | **DecoyTomcat_Enterprise0** | **扩大诱捕范围：** 部署 Tomcat 诱捕服务，目标 Enterprise0。 |
| **7** | Suspicious activity 持续 | InvestigateHost (优先级 1.1) | **InvestigateHost_Op_Host0** | **开始调查：** 在诱捕行动后，蓝队终于执行了规则推荐的高优先级调查。 |
| **8** | 回归静默 (Reward 0.0) | MonitorTraffic (优先级 0.4) | InvestigateHost_User4 | **收尾与监测：** 威胁消除，恢复主动调查状态。 |

## III. 蓝方关键决策解析：从告警到诱捕

在本局中，蓝队的决策链条在 Step 3 发生了关键的转向。虽然规则系统倾向于尽快定位威胁（调查），但蓝队策略似乎将**即时欺骗（Decoy）**置于了更高的执行权重。

以下是高优先级介入步骤的决策链条复盘：

### Step 3: 紧急欺骗响应

- **证据 Y (事实):** `Suspicious activity has been detected` 出现。
- **规则触发 (Rules):**
    - `Prefer-Investigate-On-Suspicion` (优先级 +0.7) 和 `Investigate-Priority-Over-Decoy` (优先级 +0.4) 共同将 `InvestigateHost` 推到最高优先级 (1.1)。
    - `Prefer-Decoy-Under-Suspicion` 也将 Decoy 操作的优先级提升了 0.22。
- **蓝方执行 Z 操作:** `DecoySSHD_User2`。
- **分析:** 当 Step 3 出现可疑活动证据时 → 蓝队执行了部署 SSHD 诱捕的操作。这表明蓝队选择牺牲调查时间，优先通过在敏感主机（User2）上部署诱捕服务来**分散攻击者注意力**。

### Step 4 & 6: 持续部署欺骗

- **证据 Y (事实):** `Suspicious activity has been detected` 持续存在。
- **蓝方执行 Z 操作:** `DecoySmss_Enterprise1` (Step 4) 和 `DecoyTomcat_Enterprise0` (Step 6)。
- **分析:** 面对持续的威胁，蓝队继续扩大诱捕的覆盖范围，先后在 Enterprise1 和 Enterprise0 上部署了诱捕服务。这是一种**积极防御/纵深防御**的体现，意图通过大量诱捕物分散对手的横向移动路径。

### Step 7: 延迟的调查行动

- **证据 Y (事实):** `Suspicious activity has been detected` 持续存在。
- **蓝方执行 Z 操作:** `InvestigateHost_Op_Host0`。
- **分析:** 在进行了三次诱捕部署后，蓝队终于在 Step 7 采纳了规则引擎持续推荐的调查动作，针对 Op_Host0 进行深入分析。这可能意味着蓝队认为诱捕效果已达到，或必须开始定位威胁的初始源头。

---

## IV. 规则/证据的作用与冲突

### 1. 核心触发证据

本局的核心证据是 **FACT('suspicious_activity')**，它在 Step 3 出现，并在 Step 8 前一直存在。

*   **作用：** 它是触发所有高风险响应规则的“开关”，立即屏蔽了所有低效或被动的行为（如 `Sleep`），并强制提高了调查和诱捕的优先级。

### 2. 规则冲突与实际策略

本局突出的策略冲突在于推荐优先级与实际执行之间的差异：

| 规则名称 | 条件 | 优先级/效果 | 影响 | 蓝方实际响应 |
| :--- | :--- | :--- | :--- | :--- |
| `Investigate-Priority-Over-Decoy` | Suspicion AND NOT Compromised | `InvestigateHost` +0.4 | 强烈推荐快速定位威胁。 | **被忽视。** 蓝队在 Step 3, 4, 6 持续选择 Decoy。 |
| `Prefer-Decoy-Under-Suspicion` | Suspicious activity | Decoy actions +0.22 | 鼓励使用诱捕。 | **被采用。** 蓝队将 Decoy 作为首要响应手段。 |
| `Avoid-Remove-Decoy-When-Quiet` | NOT Suspicious AND NOT Compromised | 硬屏蔽移除/诱捕 | 确保静默期不进行不必要的清理和诱捕操作。 | 在静默期（Step 1, 2, 8）有效发挥作用。 |

**观察结论：** 虽然规则引擎基于调查和定位的效率（高优先级）进行了推荐，但蓝队实际执行的策略（通过多次 Decoy 动作体现）更侧重于**“先用欺骗手段争取时间，再进行耗时的调查”**的防御哲学。PolicySpeak JSON 虽然信息有误（引述 Episode 4），但其体现的意图是明确的：基于 `suspicious_activity` 证据，执行高严重性（severity=6）的 Decoy 动作。

### V. 总结与改进建议（供 SOC 学习）

#### 1. 学习要点

*   **高优先级场景识别：** 当 `suspicious_activity` 告警触发时（Step 3），所有资源应立即转向响应，`Sleep` 等待机操作必须被硬屏蔽。
*   **Decoy 的价值：** 本局显示，即使 Investigate 优先级最高，蓝队仍优先部署 Decoy。对于 SOC 而言，这意味着在确认威胁路径不明朗时，**部署欺骗网络比盲目扫描更具战术价值**，能够有效地转移攻击者目标。

#### 2. 改进建议

1.  **明确 Decoy 目标优先级：** 蓝队在 Step 3 部署了 `DecoySSHD_User2`。未来应在策略中加入逻辑，明确哪个区域（如 Op_Server0、Enterprise 关键资产）的 Decoy 部署应优先于其他区域，避免随机性部署。
2.  **RestoreService 动作的审查：** Step 5 在可疑活动持续存在时执行了 `RestoreService_User4`。如果服务恢复不是对先前破坏的直接修正（即没有证据表明 User4 被破坏），则这种行为会浪费宝贵的时间。应在“Suspicion”规则下**降低 RestoreService 的优先级**，除非有明确证据显示服务被中断且需要恢复。
3.  **调查与欺骗的平衡：** 规则引擎强烈推荐 Investigate。如果蓝队策略是为了先欺骗，建议修改规则权重，将 Decoy 的优先级提高到与 Investigate 相近的水平，以消除规则推荐与实际行动之间的策略冲突，实现策略与规则的一致性。