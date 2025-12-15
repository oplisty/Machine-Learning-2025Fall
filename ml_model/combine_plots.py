import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

def combine_plots_to_pdf(
    train_close_path: str,
    train_volume_path: str,
    test_close_path: str,
    test_volume_path: str,
    output_pdf_path: str
):
    """
    Combine four prediction plots into a single PDF with 2x2 layout.
    
    Args:
        train_close_path: Path to train forecast close plot
        train_volume_path: Path to train forecast volume plot
        test_close_path: Path to test forecast close plot
        test_volume_path: Path to test forecast volume plot
        output_pdf_path: Output PDF file path
    """
    # Load images
    train_close_img = Image.open(train_close_path)
    train_volume_img = Image.open(train_volume_path)
    test_close_img = Image.open(test_close_path)
    test_volume_img = Image.open(test_volume_path)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('XGBoost Model Prediction Results', fontsize=16, fontweight='bold', y=0.995)
    
    # Remove axes for image display
    for ax in axes.flat:
        ax.axis('off')
    
    # Display images in 2x2 grid
    axes[0, 0].imshow(train_close_img)
    axes[0, 0].set_title('Training Set - Close Price', fontsize=12, fontweight='bold', pad=10)
    
    axes[0, 1].imshow(train_volume_img)
    axes[0, 1].set_title('Training Set - Volume', fontsize=12, fontweight='bold', pad=10)
    
    axes[1, 0].imshow(test_close_img)
    axes[1, 0].set_title('Test Set - Close Price', fontsize=12, fontweight='bold', pad=10)
    
    axes[1, 1].imshow(test_volume_img)
    axes[1, 1].set_title('Test Set - Volume', fontsize=12, fontweight='bold', pad=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save to PDF
    plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plots saved to: {output_pdf_path}")


if __name__ == "__main__":
    # Set paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output" / "xgboost" / "horizon_1"
    
    train_close_path = output_dir / "train_forecast_close.png"
    train_volume_path = output_dir / "train_forecast_volume.png"
    test_close_path = output_dir / "test_forecast_close.png"
    test_volume_path = output_dir / "test_forecast_volume.png"
    output_pdf_path = output_dir / "combined_forecast_results.pdf"
    
    # Check if all files exist
    for path in [train_close_path, train_volume_path, test_close_path, test_volume_path]:
        if not path.exists():
            print(f"Error: File not found: {path}")
            exit(1)
    
    # Combine plots
    combine_plots_to_pdf(
        str(train_close_path),
        str(train_volume_path),
        str(test_close_path),
        str(test_volume_path),
        str(output_pdf_path)
    )

