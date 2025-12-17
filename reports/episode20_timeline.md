# Episode 20 时间线概览

## Step 1
- **环境事实 (facts):**
  - Recent environment reward is 0.0.
- **规则触发 (rules):**
  - Rule 'Avoid-Remove-Decoy-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') is active and applies a hard_mask on actions ['RemoveMalware', 'RemoveMalware_Defender', 'RemoveMalware_Enterprise0', 'RemoveMalware_Enterprise1', 'RemoveMalware_Enterprise2', 'RemoveMalware_Op_Host0', 'RemoveMalware_Op_Host1', 'RemoveMalware_Op_Host2', 'RemoveMalware_Op_Server0', 'RemoveMalware_User0', 'RemoveMalware_User1', 'RemoveMalware_User2', 'RemoveMalware_User3', 'RemoveMalware_User4', 'DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'].
  - Rule 'Prefer-Monitor-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['MonitorTraffic'] by 0.4.
  - Rule 'Deprioritize-Restore-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['RestoreService', 'RestoreService_Enterprise0', 'RestoreService_Enterprise1', 'RestoreService_Enterprise2', 'RestoreService_Op_Server0'] by -0.4.
- **推荐动作 (recommended_actions):**
  - Recommended action from prior rules: MonitorTraffic with priority=0.4000000059604645.
- **蓝方实际动作 (blue_action):**
  - At episode 20, step 1, blue executed action InvestigateHost_Enterprise1 (idx=4).

## Step 2
- **环境事实 (facts):**
  - Recent environment reward is 0.0.
- **规则触发 (rules):**
  - Rule 'Avoid-Remove-Decoy-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') is active and applies a hard_mask on actions ['RemoveMalware', 'RemoveMalware_Defender', 'RemoveMalware_Enterprise0', 'RemoveMalware_Enterprise1', 'RemoveMalware_Enterprise2', 'RemoveMalware_Op_Host0', 'RemoveMalware_Op_Host1', 'RemoveMalware_Op_Host2', 'RemoveMalware_Op_Server0', 'RemoveMalware_User0', 'RemoveMalware_User1', 'RemoveMalware_User2', 'RemoveMalware_User3', 'RemoveMalware_User4', 'DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'].
  - Rule 'Prefer-Monitor-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['MonitorTraffic'] by 0.4.
  - Rule 'Deprioritize-Restore-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['RestoreService', 'RestoreService_Enterprise0', 'RestoreService_Enterprise1', 'RestoreService_Enterprise2', 'RestoreService_Op_Server0'] by -0.4.
- **推荐动作 (recommended_actions):**
  - Recommended action from prior rules: MonitorTraffic with priority=0.4000000059604645.
- **蓝方实际动作 (blue_action):**
  - At episode 20, step 2, blue executed action InvestigateHost_Op_Server0 (idx=9).

## Step 3
- **环境事实 (facts):**
  - Suspicious activity has been detected (episode=20, step=3).
  - At least one host has been discovered by Blue.
  - Recent environment reward is 0.0.
- **规则触发 (rules):**
  - Rule 'No-Sleep-Under-Suspicion' with condition FACT('suspicious_activity') OR FACT('host_compromised') OR FACT('high_risk_state') is active and applies a hard_mask on actions ['Sleep'].
  - Rule 'Prefer-Investigate-On-Suspicion' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.7.
  - Rule 'Prefer-Decoy-Under-Suspicion' with condition FACT('suspicious_activity') increases priority of actions ['DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'] by 0.22.
  - Rule 'Investigate-Priority-Over-Decoy' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.4.
- **推荐动作 (recommended_actions):**
  - Recommended action from prior rules: InvestigateHost_User4 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_Op_Host1 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_UnknownHost with priority=1.100000023841858.
- **蓝方实际动作 (blue_action):**
  - At episode 20, step 3, blue executed action DecoySSHD_User2 (idx=108).

## Step 4
- **环境事实 (facts):**
  - Suspicious activity has been detected (episode=20, step=4).
  - At least one host has been discovered by Blue.
  - Recent environment reward is 0.0.
- **规则触发 (rules):**
  - Rule 'No-Sleep-Under-Suspicion' with condition FACT('suspicious_activity') OR FACT('host_compromised') OR FACT('high_risk_state') is active and applies a hard_mask on actions ['Sleep'].
  - Rule 'Prefer-Investigate-On-Suspicion' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.7.
  - Rule 'Prefer-Decoy-Under-Suspicion' with condition FACT('suspicious_activity') increases priority of actions ['DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'] by 0.22.
  - Rule 'Investigate-Priority-Over-Decoy' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.4.
- **推荐动作 (recommended_actions):**
  - Recommended action from prior rules: InvestigateHost_User4 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_Op_Host1 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_UnknownHost with priority=1.100000023841858.
- **蓝方实际动作 (blue_action):**
  - At episode 20, step 4, blue executed action DecoySmss_Enterprise1 (idx=84).

## Step 5
- **环境事实 (facts):**
  - Suspicious activity has been detected (episode=20, step=5).
  - At least one host has been discovered by Blue.
  - Recent environment reward is 0.0.
- **规则触发 (rules):**
  - Rule 'No-Sleep-Under-Suspicion' with condition FACT('suspicious_activity') OR FACT('host_compromised') OR FACT('high_risk_state') is active and applies a hard_mask on actions ['Sleep'].
  - Rule 'Prefer-Investigate-On-Suspicion' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.7.
  - Rule 'Prefer-Decoy-Under-Suspicion' with condition FACT('suspicious_activity') increases priority of actions ['DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'] by 0.22.
  - Rule 'Investigate-Priority-Over-Decoy' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.4.
- **推荐动作 (recommended_actions):**
  - Recommended action from prior rules: InvestigateHost_User4 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_Op_Host1 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_UnknownHost with priority=1.100000023841858.
- **蓝方实际动作 (blue_action):**
  - At episode 20, step 5, blue executed action RestoreService_User4 (idx=174).

## Step 6
- **环境事实 (facts):**
  - Suspicious activity has been detected (episode=20, step=6).
  - At least one host has been discovered by Blue.
  - Recent environment reward is 0.0.
- **规则触发 (rules):**
  - Rule 'No-Sleep-Under-Suspicion' with condition FACT('suspicious_activity') OR FACT('host_compromised') OR FACT('high_risk_state') is active and applies a hard_mask on actions ['Sleep'].
  - Rule 'Prefer-Investigate-On-Suspicion' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.7.
  - Rule 'Prefer-Decoy-Under-Suspicion' with condition FACT('suspicious_activity') increases priority of actions ['DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'] by 0.22.
  - Rule 'Investigate-Priority-Over-Decoy' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.4.
- **推荐动作 (recommended_actions):**
  - Recommended action from prior rules: InvestigateHost_User4 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_Op_Host1 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_UnknownHost with priority=1.100000023841858.
- **蓝方实际动作 (blue_action):**
  - At episode 20, step 6, blue executed action DecoyTomcat_Enterprise0 (idx=131).

## Step 7
- **环境事实 (facts):**
  - Suspicious activity has been detected (episode=20, step=7).
  - At least one host has been discovered by Blue.
  - Recent environment reward is 0.0.
- **规则触发 (rules):**
  - Rule 'No-Sleep-Under-Suspicion' with condition FACT('suspicious_activity') OR FACT('host_compromised') OR FACT('high_risk_state') is active and applies a hard_mask on actions ['Sleep'].
  - Rule 'Prefer-Investigate-On-Suspicion' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.7.
  - Rule 'Prefer-Decoy-Under-Suspicion' with condition FACT('suspicious_activity') increases priority of actions ['DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'] by 0.22.
  - Rule 'Investigate-Priority-Over-Decoy' with condition FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['InvestigateHost'] by 0.4.
- **推荐动作 (recommended_actions):**
  - Recommended action from prior rules: InvestigateHost_User4 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_Op_Host1 with priority=1.100000023841858.
  - Recommended action from prior rules: InvestigateHost_UnknownHost with priority=1.100000023841858.
- **蓝方实际动作 (blue_action):**
  - At episode 20, step 7, blue executed action InvestigateHost_Op_Host0 (idx=6).

## Step 8
- **环境事实 (facts):**
  - Recent environment reward is 0.0.
- **规则触发 (rules):**
  - Rule 'Avoid-Remove-Decoy-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') is active and applies a hard_mask on actions ['RemoveMalware', 'RemoveMalware_Defender', 'RemoveMalware_Enterprise0', 'RemoveMalware_Enterprise1', 'RemoveMalware_Enterprise2', 'RemoveMalware_Op_Host0', 'RemoveMalware_Op_Host1', 'RemoveMalware_Op_Host2', 'RemoveMalware_Op_Server0', 'RemoveMalware_User0', 'RemoveMalware_User1', 'RemoveMalware_User2', 'RemoveMalware_User3', 'RemoveMalware_User4', 'DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss', 'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'].
  - Rule 'Prefer-Monitor-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['MonitorTraffic'] by 0.4.
  - Rule 'Deprioritize-Restore-When-Quiet' with condition NOT FACT('suspicious_activity') AND NOT FACT('host_compromised') increases priority of actions ['RestoreService', 'RestoreService_Enterprise0', 'RestoreService_Enterprise1', 'RestoreService_Enterprise2', 'RestoreService_Op_Server0'] by -0.4.
- **推荐动作 (recommended_actions):**
  - Recommended action from prior rules: MonitorTraffic with priority=0.4000000059604645.
- **蓝方实际动作 (blue_action):**
  - At episode 20, step 8, blue executed action InvestigateHost_User4 (idx=14).
