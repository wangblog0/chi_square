# GitHub Actions 自动发布到 PyPI

## 配置步骤

### 1. 获取 PyPI API Token

1. 登录 [PyPI](https://pypi.org/)
2. 进入账户设置：https://pypi.org/manage/account/
3. 滚动到 "API tokens" 部分
4. 点击 "Add API token"
5. Token 名称：`GitHub Actions genetics1003`
6. 作用域选择：`Entire account` 或特定项目 `genetics1003`
7. 创建后**立即复制**token（只显示一次）

### 2. 在 GitHub 仓库添加 Secret

1. 进入你的 GitHub 仓库
2. 点击 `Settings` -> `Secrets and variables` -> `Actions`
3. 点击 `New repository secret`
4. 配置如下：
   - **Name**: `PYPI_API_TOKEN`
   - **Secret**: 粘贴刚才复制的 PyPI API token
5. 点击 `Add secret`

### 3. 使用方法

#### 创建新版本并发布：

```bash
# 1. 更新版本号（在 pyproject.toml 中修改 version）
# 例如：version = "2.1.1"

# 2. 提交更改
git add .
git commit -m "Release version 2.1.1"

# 3. 创建并推送 tag（会自动触发 GitHub Action）
git tag v2.1.1
git push origin main
git push origin v2.1.1
```

#### 或者一键操作：

```bash
# 创建 tag 并推送（会自动触发发布）
git tag v2.1.1 -m "Release version 2.1.1"
git push origin v2.1.1
```

### 4. 工作流说明

- **触发条件**：推送以 `v` 开头的 tag（如 `v2.1.0`、`v3.0.0`）
- **执行步骤**：
  1. 检出代码
  2. 设置 Python 环境
  3. 安装构建工具（build、twine）
  4. 构建分发包（wheel 和 tar.gz）
  5. 使用 API token 上传到 PyPI

### 5. 查看执行状态

1. 推送 tag 后，访问仓库的 `Actions` 标签页
2. 查看 "Publish to PyPI" 工作流运行状态
3. 如果失败，点击查看详细日志排查问题

### 6. 常见问题

**Q: 如何删除错误的 tag？**
```bash
# 删除本地 tag
git tag -d v2.1.0

# 删除远程 tag
git push origin :refs/tags/v2.1.0
```

**Q: 如何重新上传同一版本？**
- PyPI 不允许重新上传相同版本号
- 需要增加版本号（如 2.1.0 -> 2.1.1）

**Q: 工作流失败怎么办？**
- 检查 `PYPI_API_TOKEN` 是否正确设置
- 确认 token 权限足够（建议使用 project-scoped token）
- 查看 Actions 日志获取详细错误信息

### 7. 版本号规范建议

遵循语义化版本控制（Semantic Versioning）：

- `v1.0.0` - 主版本号.次版本号.修订号
- **主版本号**：不兼容的 API 修改
- **次版本号**：向下兼容的功能性新增
- **修订号**：向下兼容的问题修正

示例：
- `v2.1.0` - 当前版本
- `v2.1.1` - Bug 修复
- `v2.2.0` - 新功能
- `v3.0.0` - 重大更新（可能破坏兼容性）

