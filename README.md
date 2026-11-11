# CCCD Scanner – tự nhận diện, cắt và nắn thẳng ảnh căn cước

## Cài đặt (1 lần)
```
pip install opencv-python numpy
```

## Dùng
- **Kéo-thả** ảnh (hoặc thư mục ảnh) lên file `chay_cccd.bat` (Windows) → kết quả nằm trong thư mục `output`.
- Hoặc dòng lệnh:
```
python cccd_scanner.py anh1.jpg anh2.png
python cccd_scanner.py thu_muc_anh -o ket_qua
python cccd_scanner.py anh.jpg --debug      # lưu thêm ảnh đánh dấu 4 góc
```
Tuỳ chọn: `--width 1200` (ép chiều rộng), `--enhance` (tăng nét), `--png`, `--no-orient` (không tự xoay 180°).

Kết quả: `<tên>_cccd.jpg` – thẻ nằm ngang, đã nắn thẳng, cắt gọn, tự xoay đúng chiều (mặt trước).

## Dùng trong code
```python
from cccd_scanner import scan_card, imread, imwrite
res = scan_card(imread("anh.jpg"))
if res["found"]:
    imwrite("out.jpg", res["card"])
```

## Kiểm thử
`python run_tests.py` tạo 20 ảnh thử (nền gỗ/trắng/tối/jeans/lộn xộn…, xoay 0–270°, nghiêng, mờ, thiếu sáng, loá, ngón tay che góc)
và đo độ chính xác so với đáp án đúng. Ảnh tổng hợp ở `results/contact_sheet_*.jpg`.

## Giới hạn đã biết
- Thẻ **trắng/nhạt đặt trên nền trắng** (mép thẻ gần như không phân biệt được): có thể cắt lẹm ~5% một cạnh.
- Nền **rất lộn xộn nhiều màu xanh/lục**: có thể nhận sai; khi đó chương trình in `[CANH BAO ...]`.
- Tự xoay đúng chiều dựa vào QR + quốc huy đỏ của **mặt trước**; mặt sau vẫn được cắt/nắn thẳng nhưng không tự xác định đảo đầu.
"# CCCD" 
