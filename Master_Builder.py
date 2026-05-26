import os
import subprocess
import sys
import time

# ==========================================
# 系統環境變數與防禦性初始化 (廠長級算圖模式)
# ==========================================
UE_CMD_PATH = r"c:\UE_5.3\Engine\Binaries\Win64\UnrealEditor.exe"
PROJECT_PATH = r"C:\GPS_Gaussian\GPS_Gaussian.uproject"
LOG_PATH = r"C:\GPS_Gaussian\Saved\Logs\GPS_Gaussian.log"
OUTPUT_DIR = r"D:\04.09\VideoPose3D\Renders" 

MAP_PATH = "/Game/Main" 
SEQUENCE_PATH = "/Game/RenderPeople/VideoPoseSyntheticModel/Ch01_nonPBR/Seq_Vaccine_01.Seq_Vaccine_01"
CONFIG_PATH = "/Game/RenderPeople/VideoPoseSyntheticModel/Ch01_nonPBR/MRQ_Preset.MRQ_Preset"

def setup_environment():
    """架構師級的環境淨化與防呆機制"""
    print("[*] 執行系統環境檢查...")
    
    if os.path.exists(LOG_PATH):
        try:
            os.remove(LOG_PATH)
        except PermissionError:
            print("\n[!] 🛑 致命錯誤：您的 UE5 編輯器尚未關閉！")
            print("[!] 系統拒絕執行：請先將虛幻引擎徹底關閉，再重新執行腳本。")
            sys.exit(1) 
            
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] 已確保實體影像輸出目錄存在: {OUTPUT_DIR}")

def extract_high_value_tokens(log_path):
    """【AI 對齊協定】：極低 Token 消耗的日誌特徵提取器"""
    if not os.path.exists(log_path):
        return "[!] 找不到實體日誌。"

    signal_keywords = ["Error:", "Fatal error:", "Exception", "LogMovieRenderPipeline:"]
    noise_keywords = ["LogSlate:", "LogAudio:", "Took", "LogD3D12RHI:", "Display:"]
    
    extracted_signals = []
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
            for line in lines:
                if any(noise in line for noise in noise_keywords):
                    continue
                if any(keyword in line for keyword in signal_keywords):
                    extracted_signals.append(line.strip())

        final_output = []
        final_output.append("=== 高純度訊號提取 (Token Optimized) ===")
        final_output.extend(extracted_signals[-15:])
        final_output.append("--- 日誌物理末端 ---")
        final_output.extend([line.strip() for line in lines[-5:]] if len(lines) >= 5 else lines)
        
        return "\n".join(final_output)
    except Exception as e:
        return f"[!] 提取日誌特徵失敗: {e}"

def trigger_headless_render():
    print("[*] 啟動引擎 (廠長模式 + 防阻塞監聽 + 沒收視窗)...")
    start_time = time.time()
    
    cmd = [
        UE_CMD_PATH,
        PROJECT_PATH,
        MAP_PATH,
        "-game",
        f"-LevelSequence={SEQUENCE_PATH}",
        f"-MoviePipelineConfig={CONFIG_PATH}",
        "-RenderOffScreen",
        "-resx=1000", "-resy=1000",
        "-unattended",
        "-nopause",
        "-NoSplash",
        "-NoLiveCoding", 
        "-d3d11",
        "-StdOut",
        "-FullStdOutLogOutput"
    ]
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )
        
        print("\n=== UE5 即時日誌串流開始 ===")
        print("【系統提示】：若文字停止跳動，為底層 Shader 冷編譯，請『靜置 15 分鐘』切勿強殺視窗！")
        
        while True:
            output = process.stdout.read(1)
            if output == '' and process.poll() is not None:
                break
            if output:
                sys.stdout.write(output)
                sys.stdout.flush()
                
        print("\n=== UE5 即時日誌串流結束 ===\n")
        
        if process.returncode == 0:
            print(f"[*] 算圖任務完成，耗時: {time.time() - start_time:.2f} 秒")
        else:
            print(f"\n[!] 【算圖異常】錯誤碼: {process.returncode}")
            print(extract_high_value_tokens(LOG_PATH))
            print("=============================================\n")
            
    except Exception as e:
        print(f"[!] 系統級崩潰: {e}")

def pipeline_orchestrator():
    print("========================================")
    print(" 啟動 V167 無頭渲染產線 (僅執行影像輸出)")
    print("========================================")
    
    setup_environment()
    trigger_headless_render()
    
    # ⚠️ 已物理抹除 clean_and_extract_log() 呼叫
    
    print("========================================")
    print(" 算圖管線執行完畢，請進行 2D 矩陣降維。")
    print("========================================")

if __name__ == "__main__":
    pipeline_orchestrator()