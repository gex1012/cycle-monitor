# 部署到 Streamlit Community Cloud（免费 · 永久网址）

部署后会得到一个形如 `https://你的应用名.streamlit.app` 的网址，**任何人点开即用，你的电脑无需开机**。

> 我已经帮你把项目初始化成 git 仓库并完成首次提交，下面只剩 3 步：建仓库 → 推送 → 部署。

---

## 第 1 步：在 GitHub 建一个空仓库
1. 打开 <https://github.com> ，注册/登录（用你的邮箱即可）。
2. 右上角 **+ → New repository**。
3. 仓库名随意，例如 `cycle-monitor`；可见性选 **Private（私有）或 Public 都行**（私有也能部署）。
4. **不要**勾选 “Add a README / .gitignore / license”（保持空仓库）。
5. 点 **Create repository**，记下页面给出的地址：`https://github.com/你的用户名/cycle-monitor.git`

## 第 2 步：把代码推上去
在本项目目录（`G:\My Drive\Model`）打开 PowerShell，依次执行（把地址换成你自己的）：

```powershell
git remote add origin https://github.com/你的用户名/cycle-monitor.git
git branch -M main
git push -u origin main
```

- 第一次推送会弹出登录窗口：用浏览器授权 GitHub 即可（或粘贴一个 Personal Access Token 作为密码）。
- 看到 `main -> main` 即推送成功。

## 第 3 步：在 Streamlit 上一键部署
1. 打开 <https://share.streamlit.io> ，点 **Sign in with GitHub** 并授权。
2. 点 **Create app / New app → Deploy a public app from GitHub**。
3. 填三项：
   - **Repository**：选你刚建的 `cycle-monitor`
   - **Branch**：`main`
   - **Main file path**：`app.py`
4. （可选）**Advanced settings → Python version** 选 `3.13`（不选也能跑）。
5. 点 **Deploy**，等 1–3 分钟装依赖。完成后得到网址，复制发给任何人即可。

---

## 之后怎么维护
- **更新内容**：本地改完代码后执行 `git add -A && git commit -m "更新" && git push`，Streamlit 会**自动重新部署**。
- **数据**：App 实时从 Yahoo Finance / FRED 抓取，无需上传任何数据文件。
- **偶发抓取失败**：云端 IP 偶尔被 Yahoo 限流，某个标的显示“数据缺失”时点侧栏 **🔄 刷新行情** 重试即可（FRED 宏观不受影响）。
- **访问控制**：默认任何拿到网址的人都能访问。需要限制时，可在 Streamlit 应用的 **Settings → Sharing** 里设为仅指定邮箱可看。
- **休眠**：免费版应用长时间无人访问会休眠，下次有人打开会自动唤醒（首屏稍慢几秒）。

## 不需要的东西
- 本项目**没有任何密钥/secrets**，无需配置。
- `requirements.txt` 已列好依赖，Streamlit 云会自动安装，你无需手动操作。
