# Doctor Language Bridge

面向医生的多语言图文问答网页，流程如下：

- 输入文字和可选图片
- 先让模型判断文字是否为英语
- 英语：直接进入英文回答
- 非英语：先判断语种并翻译成英文，再基于英文问题和图片回答
- 最后把英文回答回译回原始语种
- 回答阶段启用 `logprobs`，在前端展示 token 级置信度

## 运行

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

```bash
export OPENROUTER_MODEL="openai/gpt-4o"
```

3. 启动服务

```bash
uvicorn doctor_web.app:app --reload --port 8000
```

4. 打开浏览器

```text
http://127.0.0.1:8000
```

5. 在网页里输入你自己的 OpenRouter API key

- 先去 `https://openrouter.ai/` 注册账户
- 在 OpenRouter 后台创建你自己的 API key
- 把 key 粘贴到网页里的 `Your OpenRouter API key` 输入框后再运行分析

当前站点不会使用服务端持有者的 OpenRouter key；每个用户都需要自己提供 key。

## 可调参数

- `OPENROUTER_MODEL`：默认 `openai/gpt-4o`
- `OPENROUTER_PORTAL_URL`：默认 `https://openrouter.ai/`
- `DOCTOR_WEB_TOP_LOGPROBS`：默认 `5`
- `DOCTOR_WEB_MAX_INPUT_CHARS`：默认 `12000`
- `DOCTOR_WEB_MAX_IMAGE_COUNT`：默认 `4`

## 公网部署

如果你希望任何人都可以访问这个网站，最直接的方式是把当前 FastAPI 服务部署到公网，然后绑定一个域名。

### 方案 A：Render

仓库根目录已经提供了 [render.yaml](/mnt/data3/yuqian/multilingual/render.yaml)。

1. 把项目推到 GitHub
2. 在 Render 里选择 `New -> Blueprint`
3. 连接你的 GitHub 仓库
4. 导入 `render.yaml`
5. 部署完成后，把你的公网域名写到：
   - `OPENROUTER_SITE_URL`
### 方案 B：Railway / Fly.io / 自己的服务器

仓库根目录已经提供了 [Dockerfile](/mnt/data3/yuqian/multilingual/Dockerfile)，可直接用于 Docker 部署。

容器启动命令等价于：

```bash
uvicorn doctor_web.app:app --host 0.0.0.0 --port $PORT
```

### 必须注意

当前网站已经改成“用户自己提供 OpenRouter key”的模式。

也就是说：

- 站点部署者不需要在服务端配置 `OPENROUTER_API_KEY`
- 每个访问者都需要先注册 OpenRouter，并填写自己的 key
- 请求仍然会经过你的服务端转发，所以隐私说明和访问控制仍然很重要

如果你要真正对外开放，仍然建议补下面几项：

- 访问控制：登录、邀请码或机构邮箱白名单
- 频率限制：按 IP 限制请求频率
- 防滥用：Cloudflare Turnstile / CAPTCHA
- 隐私声明：如果用户会上传医疗图片，这一点尤其重要
