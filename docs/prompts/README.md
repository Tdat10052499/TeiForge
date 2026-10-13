# Sổ tay prompt thực thi TeiForge với Claude Code

Bộ prompt này dùng với Claude Code CLI chạy trong terminal của VS Code. Phần lớn đã được đóng
gói thành lệnh slash trong `.claude/skills/`, nên bạn chỉ cần gõ `/tên-lệnh`. Các quy ước chung
của dự án nằm trong `CLAUDE.md` và được nạp tự động ở mỗi phiên.

## 1. Danh sách lệnh

| Lệnh | Khi nào dùng | Có sửa mã? | Có thao tác không hoàn tác được? |
| --- | --- | --- | --- |
| `/tf-onboard` | Lần đầu trên một máy, hoặc sau khi pull nhiều thay đổi | Không (chỉ tạo `.venv`) | Không |
| `/tf-check` | Trước mỗi commit; khi CI đỏ | Có (sửa lỗi lint/test) | Không |
| `/tf-check --report-only` | Chỉ muốn xem trạng thái | Không | Không |
| `/tf-smoke-llm openai` | Kiểm thử với API thật (tốn phí) | Không | Không (có gọi API trả phí, hỏi trước) |
| `/tf-release 0.3.0` | Phát hành một phiên bản | Không | **Có**: merge, tag, publish; mỗi bước hỏi riêng |
| `/tf-plan-next 0.4` | Lập kế hoạch phiên bản tiếp theo | Chỉ viết file kế hoạch | Không |
| `/tf-feature P2` | Triển khai một hạng mục trong kế hoạch | Có, trên nhánh riêng | Push và mở draft PR (hỏi trước) |
| `/tf-review` hoặc `/tf-review 12` | Review nhánh hiện tại hoặc PR số 12 | Không | Không |

Các lệnh đều đặt `disable-model-invocation: true` (trừ `/tf-check`), nghĩa là Claude không tự
chạy chúng; chỉ bạn gọi.

## 2. Lộ trình thực thi

### Giai đoạn A: hoàn tất và phát hành v0.3

```text
/tf-onboard
/tf-smoke-llm openai          # hoặc anthropic / ollama, tùy key bạn có
/tf-review 1                  # review lần cuối PR #1
/tf-release 0.3.0
```

Trước `/tf-release`, bạn cần tự làm trên web (Claude không làm thay được):

1. PyPI: tạo Trusted Publisher cho dự án `teiforge` (hướng dẫn ở đầu `.github/workflows/release.yml`).
2. GitHub: Settings → Environments → tạo environment tên `pypi`.

### Giai đoạn B: v0.4

```text
/tf-plan-next 0.4
```

Đọc và chỉnh `docs/plans/v0.4-plan.md`, sau đó triển khai lần lượt từng hạng mục:

```text
/tf-feature P1
/tf-review
/tf-check
```

Lặp lại cho P2, P3, ... Mỗi hạng mục tương ứng một PR. Merge xong mới làm hạng mục tiếp theo,
để các PR không chồng chéo.

## 3. Prompt soạn sẵn cho các tình huống thường gặp

Dán trực tiếp vào phiên `claude`. Các đoạn trong `<...>` cần thay bằng thông tin thật.

### 3.1. CI đỏ trên một PR

```text
CI của PR #<số> đang đỏ. Đọc log của job lỗi (qua GitHub MCP hoặc `gh run view --log-failed`),
xác định nguyên nhân gốc, phân biệt lỗi do mã với lỗi do môi trường CI (ví dụ chỉ hỏng trên
Windows). Viết test hồi quy tái hiện lỗi nếu là lỗi mã, sửa, chạy /tf-check, rồi đề xuất commit.
Chưa push khi tôi chưa đồng ý.
```

### 3.2. Xử lý một issue người dùng báo

```text
Đọc issue #<số> trên repo TeiForge. Tóm tắt bằng tiếng Việt: hành vi mong đợi, hành vi thực
tế, môi trường. Tái hiện lỗi bằng một test offline (không gọi mạng, không gọi LLM thật). Nếu
không tái hiện được, nói rõ và liệt kê thông tin còn thiếu để hỏi lại người báo. Nếu tái hiện
được, đề xuất cách sửa và chờ tôi duyệt.
```

### 3.3. Thêm một tool mới vào bộ built-in

```text
Đề xuất thêm tool built-in `<tên>` để <mục đích>. Trước khi viết mã, phân tích:
(1) tool có cần mạng hoặc truy cập tệp không, và rủi ro bảo mật theo SECURITY.md;
(2) nên nằm trong core hay một optional extra;
(3) schema tham số và các trường hợp lỗi (ToolError).
Sau khi tôi duyệt, triển khai theo quy trình của /tf-feature.
```

### 3.4. Thêm một LLM provider mới

```text
Thêm provider `<tên>` vào teiforge/llm.py theo giao thức LLMProvider. Yêu cầu: SDK nằm trong
optional extra riêng; chuyển đổi định dạng message trung lập (tool_calls, role "tool",
is_error) sang API của provider; chuẩn hóa lỗi thành LLMError; test bằng client giả lập giống
tests/test_llm.py, không gọi mạng. Kiểm tra tài liệu chính thức của provider cho tên tham số,
ghi rõ phần nào chưa xác minh được. Trình bày thiết kế trước khi viết mã.
```

### 3.5. Kiểm tra sức khỏe dự án định kỳ

```text
Đánh giá tình trạng TeiForge (chỉ đọc): kết quả /tf-check --report-only; các dependency có bản
mới hoặc cảnh báo bảo mật; issue và PR đang mở quá 14 ngày; mục chưa tick trong kế hoạch hiện
tại; phần mã có coverage thấp nhất. Trình bày theo cấu trúc Bối cảnh → Phát hiện → Đề xuất ưu tiên.
```

### 3.6. Chuẩn bị bài giới thiệu dự án

```text
Dựa trên README.md và CHANGELOG.md, viết bản nháp bài giới thiệu TeiForge <phiên bản> (tiếng
Anh, ~300 từ) cho GitHub Discussions và Reddit r/Python: vấn đề dự án giải quyết, điểm khác
biệt có bằng chứng (sandbox, router offline, event stream), ví dụ 10 dòng mã chạy được, và
giới hạn hiện tại. Không phóng đại; mọi tuyên bố phải kiểm chứng được trong repo.
```

## 4. Nguyên tắc khi viết prompt mới

- **Nêu rõ phạm vi**: chỉ đọc, hay được sửa mã; được push hay không.
- **Đòi bằng chứng**: yêu cầu trích tệp/dòng, kết quả lệnh đã chạy; phân biệt điều đã kiểm chứng
  với giả định.
- **Một PR một việc**: chia nhỏ theo hạng mục kế hoạch.
- **Không đưa bí mật vào prompt**: API key chỉ nằm trong biến môi trường.
- Khi một prompt được dùng lặp lại từ lần thứ ba, hãy đóng gói nó thành skill mới trong
  `.claude/skills/<tên>/SKILL.md`.
