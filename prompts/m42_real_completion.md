# Prompt: 真正完成 M4.2

## 目标

把 benchmark 环境搭起来，并把口径限制在“环境是否可用、可追溯、可复现”。

## 这轮工作应该完成什么

1. benchmark registry 与 bundle 体系
   - manifest 校验
   - smoke fixture 与 acceptance-grade external bundle 分离
   - provenance 可追溯

2. adapter 与 protocol
   - Confidence Database
   - Iowa Gambling Task
   - 现有可复用的 control task
   - 统一 observation / action / feedback / confidence / trace export 接口

3. 环境级 acceptance artifacts
   - protocol artifact
   - trace export
   - replay / reproducibility artifact
   - leakage report
   - acceptance report

4. 环境级 honesty
   - 缺 bundle 就 block
   - repo fixture 只能是 smoke-only
   - 不把环境跑通误写成 behavioral fit 成立

## 明确不要做的事

- 不要把 `M4.2` 写成 human-alignment 已经成立
- 不要把 baseline comparison 塞进 `M4.2`
- 不要把 latent parameter identifiability 塞进 `M4.2`
- 不要把 benchmark quality 与 non-circular fit 问题假装已经解决

这些属于 `M4.3`。

## 输出要求

- 文档口径必须把 `M4.2` 写成 benchmark environment layer
- acceptance gates 以 registry、schema、adapter、replay、leakage、report honesty 为主
- 如果提到 benchmark quality、fit、baseline，必须明确“deferred to M4.3”
