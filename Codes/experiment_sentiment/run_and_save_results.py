#!/usr/bin/env python3
"""
Run training and save results to file.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

def run_training_and_save():
    """Run training demonstration and save results to file."""
    print("Starting training and saving results...")
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"training_results_{timestamp}.txt"
    
    # Capture all output
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # Create string buffer to capture output
    output_buffer = io.StringIO()
    
    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            # Import and run the final results
            from final_results import show_final_results
            show_final_results()
        
        # Get the captured output
        output_content = output_buffer.getvalue()
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"Results saved to: {output_file}")
        
        # Also display the results
        print("\n" + "="*70)
        print("TRAINING RESULTS")
        print("="*70)
        print(output_content)
        
        return True, str(output_file)
        
    except Exception as e:
        error_msg = f"Error running training: {e}"
        print(error_msg)
        
        # Save error to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)
            f.write("\n\nTraceback:\n")
            import traceback
            f.write(traceback.format_exc())
        
        return False, str(output_file)

def main():
    """Main function."""
    print("Modular Pseudo-Civility Detection System")
    print("=" * 60)
    print("Running training and saving results to file...\n")
    
    success, output_file = run_training_and_save()
    
    if success:
        print(f"\n[SUCCESS] Training completed successfully!")
        print(f"[INFO] Results saved to: {output_file}")
        print(f"[INFO] Results directory: {Path(output_file).parent}")
        return 0
    else:
        print(f"\n[ERROR] Training failed!")
        print(f"[INFO] Error details saved to: {output_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
