# 对齐检查 v1.a 结论表

- 模型：`openbmb/MiniCPM5-2B@12a3808a956f869c767195e9266b59c4d21d92e2`（钉定 commit）
- 环境：python 3.14.3 · transformers 5.6.2 · tokenizers-only / 零 GPU · darwin

| 检查 | 结果 | 证据（节选） | revision |
|---|---|---|---|
| C1 来源锁定 | PASS | openbmb/MiniCPM5-2B@12a3808a956f 加载成功（离线：以缓存为准） | `12a3808a956f` |
| C2 特殊token映射 | PASS | </s>→1, <\|im_end\|>→130073, <s>→0（config.json: eos_token_id=[1, 130073], bos=0） | `12a3808a956f` |
| C3 模板渲染·关思考 | PASS | 3 样例头部均 '<s>'（post_processor 前置 BOS，id 0）、末尾均 '<\|im_start\|>assistant\n<think>\n\n</think>\n\n'；含 system 指令 | `12a3808a956f` |
| C4 字母tokenization | PASS | 24/24 字母单独编码均 1 token；拼接渲染文本后无边界合并；'A'→[54]；渲染文本头部含 post_processor 前置的单个 <s>（id 0），encode(add_special_tokens=False) 不再二次加 BOS——训练加载必须 add_special_tokens=False，防双 BOS | `12a3808a956f` |
| C5 loss mask ±0 | PASS | 监督 span=[179,180] decode='A<\|im_end\|>'；首='A' 尾='<\|im_end\|>'；设计：监督到 <\|im_end\|>，轮间 \n 不入监督 | `12a3808a956f` |
| C6 结束标记三处 | PASS | tokenizer.eos='</s>'(id1)｜模板 assistant 结束=<\|im_end\|>(id130073)｜generation_config.eos=[1,130073]；训练用 <\|im_end\|>，禁用 eos_token 机械拼接；generation_config 默认 do_sample/T=1.0——推理须显式 temperature=0 | `12a3808a956f` |
| C7 解析对抗 | PASS | 10/10 对抗样例符合预期（含大小写/多字母/空/解释文本/思考残留） | `12a3808a956f` |
