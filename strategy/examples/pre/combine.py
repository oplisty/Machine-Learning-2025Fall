import os
import fitz  # PyMuPDF


# =========================
# 配置：输入 PDF 和输出文件
# =========================
NAV = "report_nav_curve.pdf"
BOTTOM = [
    "report_drawdown_curve.pdf",
    "report_position_curve.pdf",
    "report_score_thresholds.pdf",
    "report_quantile_return.pdf",
]
OUT = "report_all_figures_1page.pdf"


def must_exist(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {os.path.abspath(path)}")


def place_pdf_page(dst_page: fitz.Page, pdf_path: str, target_rect: fitz.Rect):
    """把单页 PDF（第0页）以矢量方式嵌入到目标页面的指定区域"""
    src = fitz.open(pdf_path)
    try:
        dst_page.show_pdf_page(target_rect, src, pno=0, keep_proportion=True)
    finally:
        src.close()


def main():
    # =========================
    # 1) 文件存在性检查
    # =========================
    must_exist(NAV)
    for f in BOTTOM:
        must_exist(f)

    # =========================
    # 2) 页面尺寸：A4 竖版（单位 pt）
    # =========================
    page_w, page_h = fitz.paper_size("a4")

    # =========================
    # 3) 排版参数（你主要改这些）
    # =========================
    margin = 14       # 页边距（上下左右，pt；14pt≈5mm）

    gap_top = 2       # NAV 大图 与 下方四图之间（垂直间距）
    gap_row = 2       # 下方两排之间（垂直间距）
    gap_col = 3       # 下方两列之间（水平间距）

    top_h_ratio = 0.50  # 顶部 NAV 占可用高度比例（0.45~0.55 都可）

    # =========================
    # 4) 创建输出 PDF
    # =========================
    doc_out = fitz.open()
    page = doc_out.new_page(width=page_w, height=page_h)

    # 可用区域
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    # 顶部 NAV 区域高度
    top_h = usable_h * top_h_ratio

    # 下方 2x2 区域高度（扣掉 NAV 与下方的间距 gap_top）
    bottom_h = usable_h - top_h - gap_top
    if bottom_h <= 0:
        raise ValueError("bottom_h <= 0：请调小 top_h_ratio 或 margin/gap_top。")

    # 下方每个格子尺寸（两排之间 gap_row，两列之间 gap_col）
    cell_w = (usable_w - gap_col) / 2
    cell_h = (bottom_h - gap_row) / 2
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError("cell_w/cell_h <= 0：请调小 margin/gap_row/gap_col 或 top_h_ratio。")

    # =========================
    # 5) 计算每张图的目标矩形
    # =========================
    # 顶部 NAV：整行
    rect_top = fitz.Rect(
        margin,
        margin,
        margin + usable_w,
        margin + top_h
    )

    # 下方第一排 y 起点
    y0 = margin + top_h + gap_top

    # 下方两列 x 起点
    x_left = margin
    x_right = margin + cell_w + gap_col

    # 第一排（上）
    rect_11 = fitz.Rect(x_left,  y0, x_left + cell_w,  y0 + cell_h)          # 左上
    rect_12 = fitz.Rect(x_right, y0, x_right + cell_w, y0 + cell_h)         # 右上

    # 第二排（下）
    y1 = y0 + cell_h + gap_row
    rect_21 = fitz.Rect(x_left,  y1, x_left + cell_w,  y1 + cell_h)          # 左下
    rect_22 = fitz.Rect(x_right, y1, x_right + cell_w, y1 + cell_h)         # 右下

    # =========================
    # 6) 嵌入 PDF（矢量贴图）
    # =========================
    place_pdf_page(page, NAV, rect_top)
    place_pdf_page(page, BOTTOM[0], rect_11)
    place_pdf_page(page, BOTTOM[1], rect_12)
    place_pdf_page(page, BOTTOM[2], rect_21)
    place_pdf_page(page, BOTTOM[3], rect_22)

    # =========================
    # 7) 保存
    # =========================
    out_path = os.path.abspath(OUT)
    doc_out.save(out_path, deflate=True)
    doc_out.close()
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
