# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature càng tăng thì câu trả lời càng đa dạng và khó đoán: ở temp=0.0
> model luôn chọn cách trả lời gần như xác định nhất (trong thí nghiệm của tôi cả
> hai lần đều nói về hang Sơn Đoòng, một sự thật "an toàn" và nổi tiếng), còn ở
> 1.0–1.5 nội dung đổi chủ đề (xuất khẩu hạt tiêu), câu văn ngắn hơn và mang tính
> kể chuyện tự nhiên hơn. Quy luật: temperature thấp → ổn định, lặp lại được;
> temperature cao → đa dạng, sáng tạo nhưng kém tin cậy.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.2–0.3. Support ticket cần câu trả lời chính
> xác, nhất quán và có thể lặp lại giữa các khách hàng; temperature thấp giúp model
> ít bị "bịa" thông tin và không đổi giọng điệu mỗi lần. Không nên đặt 0.0 tuyệt
> đối vì câu trả lời có thể bị khô cứng, còn temperature cao (mặc định 0.7) sẽ làm
> câu trả lời lạc đề ngoài ý muốn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Khối lượng: 10.000 × 3 = 30.000 lượt/ngày × 350 token = 10,5 triệu token
> output/ngày. GPT-4o: 10.500.000 / 1000 × $0,010 = $105/ngày (~$3.150/tháng).
> GPT-4o-mini: 10.500.000 / 1000 × $0,0006 = $6,3/ngày. Vậy GPT-4o đắt hơn
> khoảng **16–17 lần** (tỷ lệ đúng bằng tỷ lệ giá output $0,010/$0,0006 ≈ 16,7).
> Nên dùng GPT-4o khi độ chính xác quyết định tiền bạc/thương hiệu: chấm điểm
> bài viết, phân tích pháp lý, sinh code phức tạp — sai sót nhỏ đắt hơn tiền API.
> Nên dùng mini cho khối lượng lớn ít nhạy cảm: tóm tắt ticket, phân loại cảm xúc,
> dịch thô, chatbot FAQ.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi khác nhau rõ rệt cả về từ vựng lẫn cấu trúc. Persona "giáo viên
> tiểu học" dùng phép so sánh gần gũi ("cuốn sổ lớn") và giải thích bằng câu
> ngắn, tránh thuật ngữ; persona "chuyên gia tài chính" dùng thuật ngữ chuyên
> ngành (Distributed Ledger Technology, nút mạng, cơ chế đồng thuận, hàm băm) và
> trả lời có đánh số, chi tiết hơn. Kết luận: system prompt định hình toàn bộ
> giọng điệu, độ sâu và khuôn mẫu trình bày — model bám theo vai trò được giao
> hơn là câu hỏi đơn thuần.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn tiếng Việt 102 từ tôi kiểm tra, ước lượng `số từ / 0,75` = 136 token
> còn `count_tokens` (tiktoken) cho 124 token — chênh khoảng **9%** (hướng thừa).
> Hiểu lầm phổ biến: tiếng Anh trung bình 1 token ≈ 0,75 từ (tức ~1,33 token/từ),
> còn tiếng Việt tiktoken thường mã hóa ~1,9–2,4 token mỗi từ dài vì dấu và
> từ ghép bị tách nhỏ hơn — nên cùng số từ, đoạn tiếng Việt tốn nhiều token hơn
> tiếng Anh. Ước lượng thô Part 1 chỉ nên dùng khi chưa có tiktoken.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi dài và người dùng đang chờ trực tiếp —
> chat, trợ lý viết nội dung, sinh code dài — vì token hiện ra dần khiến thời
> gian chờ cảm giác ngắn hơn hẳn so với chờ trọn cả khối, đặc biệt khi model to
> mất 5–15 giây để sinh xong. Nó cũng cho phép hiện phần đầu ngay và hủy giữa
> chừng khi nội dung đã đủ (tiết kiệm token). Non-streaming phù hợp hơn khi
> backend cần nguyên một phản hồi để xử lý tiếp (phân tích, lưu trữ, chấm điểm),
> khi output ngắn và cố định (kiểm tra key, bài tập dạng MCQ) thì streaming
> không thêm giá trị, và khi cần chain/tool gọi tiếp nhận kết quả trọn vẹn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff làm giảm khối lượng yêu cầu đổ vào server đang nghẽn: lần
> thử đầu dồn dập (0,1s) nhưng nếu vẫn lỗi, các client tự "khoan ra" với khoảng
> chờ tăng gấp đôi (0,2s → 0,4s → 0,8s…) nên số request tại mỗi thời điểm giảm
> dần, giúp server có thời gian hồi phục và tăng xác suất thành công. Nếu hàng
> nghìn client cùng chờ đúng một khoảng cố định (ví dụ đều 1 giây), chúng sẽ
> "dậy sóng" cùng lúc tạo hiệu ứng bầy đàn (thundering herd) — server càng tắc,
> ai cũng thử lại đúng lúc nhau, và tai nạn lặp lại sau mỗi chu kỳ delay.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt của tôi: *"Bạn là trợ giảng thân thiện của khóa AI, trả lời
> ngắn gọn bằng tiếng Việt."* Hai lựa chọn từ ngữ quan trọng: (1) **"thân thiện"**
> định hình giọng điệu khiến model xưng hô mềm mại, tránh trả lời máy móc —
> phù hợp môi trường giáo dục; (2) **"trả lời ngắn gọn bằng tiếng Việt"** chốt
> hai ràng buộc dễ đo: vì đây là trợ lý hỏi–đáp nhanh (không phải trình soạn
> văn bản dài) nên giới hạn độ dài tiết kiệm token và thời gian đọc, và chỉ định
> rõ ngôn ngữ để tránh model trả lời tiếng Anh khi prompt tiếng Việt.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là **history chỉ giữ 3 lượt gần nhất** nên trợ lý quên các
> thông tin đã trao đổi trước đó trong cùng phiên (tên người dùng, dự án đang
> làm…) và hoàn toàn không có bộ nhớ giữa các phiên. Cải thiện cụ thể: thay
> hard-code bằng chiến lược tóm tắt cuộn — cứ mỗi N lượt (ví dụ 6), lấy toàn bộ
> history cũ đưa vào một prompt tóm tắt bằng model rẻ (`OPENAI_MINI_MODEL`) để
> sinh "Bản tóm tắt hội thoại đến nay", luôn đặt đoạn tóm tắt đó vào đầu
> `messages` làm context system phụ, rồi vẫn giữ 3 lượt gần nhất để trả lời
> chính xác. Cách triển khai: một dict lưu `summary` + `history`; mỗi lượt kiểm
> tra `len(history) >= 6` thì gọi `call_openai_mini` với prompt tóm tắt, cập nhật
> summary, reset history — chi phí thấp hơn nhiều so với gửi cả lịch sử dài mỗi
> lượt mà vẫn giữ được ngữ cảnh dài hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026