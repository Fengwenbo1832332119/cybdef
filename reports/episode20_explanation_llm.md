# Episode 20 蓝队行动总结报告

## 总体态势

Episode 20 的防御行动经历了从平静到高警戒状态的快速转换。在前两个步骤中环境保持静默，但在第 3 步检测到可疑活动，使系统进入高优先级防御模式。蓝方在威胁出现后采取了多项分散的防御措施，主要集中在蜜罐部署和主机排查。本回合环境奖励为 0.0，未发现明确的主机沦陷事件。

## 威胁演进

本回合的核心事件是 **可疑活动（suspicious activity）**的出现与消退：

1. **初始静默（Steps 1-2）：** 环境平静，未报告任何异常。蓝方利用这段时间进行了对 Enterprise1 和 Op\_Server0 的主动侦查。
2. **警报触发（Steps 3-7）：** 在第 3 步，系统事实（FACT）检测到可疑活动。这一状态持续到第 7 步。尽管威胁规则被激活，但蓝方未能在本回合内将可疑活动升级为明确的主机入侵（host\_compromised）。
3. **静默恢复（Step 8）：** 在第 8 步，可疑活动的事实消失，系统规则回归到静默状态。

## 蓝方关键操作及策略分析

在整个 Episode 20 中，蓝方的行动策略显示出从主动侦查到反应式防御，再到防御性蜜罐部署的转变，但行动的集中度不足。

| 步骤 | 环境状态 | 规则推荐 (最高优先级) | 蓝方实际行动 | 偏差分析 |
| :--- | :--- | :--- | :--- | :--- |
| 1-2 | 静默 | MonitorTraffic (P=0.4) | InvestigateHost (E1, Op\_S0) | 忽略流量监控，执行本地侦查。 |
| 3-7 | **可疑活动** | InvestigateHost (P=1.1) | DecoySSHD\_User2, DecoySmss\_Enterprise1, RestoreService\_User4, DecoyTomcat\_Enterprise0, InvestigateHost\_Op\_Host0 | 持续偏离最高推荐行动。在高度警觉下，蓝方优先部署蜜罐而非快速定位威胁。Step 5 的服务恢复 (RestoreService) 尤为异常。 |
| 8 | 静默恢复 | MonitorTraffic (P=0.4) | InvestigateHost\_User4 | 在静默状态下继续执行本地侦查。 |

**核心发现：** 当系统事实报告存在可疑活动时（Steps 3-7），蓝方被规则体系明确推荐执行最高优先级的 InvestigateHost 操作（优先级 P=1.1），但在 5 个响应步骤中，蓝方有 4 次选择了部署蜜罐（Decoy）或执行不相关的恢复操作（RestoreService），显示出策略选择与规则意图存在明显的非一致性。

## 规则触发机制

本回合的规则触发清晰反映了环境状态的剧烈变化：

1. **静默规则（Steps 1, 2, 8）：**
    *   `Avoid-Remove-Decoy-When-Quiet` 规则生效，对清除恶意软件和部署蜜罐等高风险/高成本动作实施了 **硬屏蔽 (hard_mask)**。
    *   `Prefer-Monitor-When-Quiet` 规则将流量监控优先级提升至 0.4。
2. **可疑规则（Steps 3-7）：**
    *   可疑活动事实触发了防御重心转移：`No-Sleep-Under-Suspicion` 立即屏蔽了休眠（Sleep）动作。
    *   **高优先级侦查配置：** `Prefer-Investigate-On-Suspicion` (P+0.7) 和 `Investigate-Priority-Over-Decoy` (P+0.4) 共同作用，将主机侦查（InvestigateHost）的优先级提升至 1.1，使其成为首选响应动作。
    *   蜜罐部署优先级被 `Prefer-Decoy-Under-Suspicion` 提升 (P+0.22)，但仍低于侦查行动。

## 最终评估

Episode 20 中，蓝方系统对环境状态转换的感知是准确的，防御规则也成功地将优先级分配给正确的响应类型（从监控/侦查转向紧急侦查/蜜罐）。

然而，蓝方执行的动作与规则体系中的最高推荐动作存在显著偏差。在威胁出现期间，防御资源被大量投入到蜜罐部署（DecoySSHD, DecoySmss, DecoyTomcat）中，而非执行规则强烈建议的主机侦查。这种偏差可能导致威胁定位的延迟，尤其是在高度警觉状态下未能充分利用 InvestigateHost 的高优先级。总体而言，蓝方成功应对了威胁的出现并采取了积极的防御动作，但执行效率和策略聚焦性有待提高。