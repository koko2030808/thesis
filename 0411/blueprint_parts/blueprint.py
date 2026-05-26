import re
import os

def process_and_split_blueprint(input_file, clean_output, chunk_size=100):
    print(f"[*] 步驟 1: 啟動全量邏輯提取 -> {input_file}")
    
    # --- 核心清洗邏輯 ---
    logic_keywords = ['Begin Object', 'End Object', 'PinName', 'LinkedTo', 
                      'VariableReference', 'MemberName', 'FunctionReference', 
                      'DefaultValue', 'InputKey', 'MemberGuid']
    
    noise_patterns = [
        r'NodePosX=-?\d+,', r'NodePosY=-?\d+,', r'NodeGuid=[A-Z0-9\-]+,', 
        r'PersistentGuid=[A-Z0-9\-]+,', r'ExportPath=".*?",', r'PinToolTip=".*?",',
        r'PinFriendlyName=NSLOCTEXT\(.*?\),', r'bCommentBubblePinned=\w+,'
    ]

    lines = []
    # 支援多種 UE 導出編碼
    for enc in ['utf-16', 'utf-16-le', 'utf-8']:
        try:
            with open(input_file, 'r', encoding=enc) as f:
                content = f.read()
                lines = content.splitlines()
            if len(lines) > 1:
                print(f"[*] 成功識別編碼: {enc}，原始總行數 {len(lines)}")
                break
        except:
            continue

    if len(lines) <= 1:
        print("[!] 錯誤: 無法讀取檔案內容，請檢查路徑或編碼。")
        return

    cleaned_data = []
    for line in lines:
        if any(k in line for k in logic_keywords):
            temp_line = line
            for pattern in noise_patterns:
                temp_line = re.sub(pattern, '', temp_line)
            cleaned_data.append(temp_line.strip())

    # 輸出清洗後的總表 (blueprint_clean.txt)
    with open(clean_output, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(cleaned_data))
    print(f"[+] 清洗完成，提取邏輯行數: {len(cleaned_data)}")

    # --- 步驟 2: 每 100 個 End Object 切分邏輯 ---
    print(f"[*] 步驟 2: 啟動模組化切分 (每 {chunk_size} 個對象一檔)")
    
    output_dir = "blueprint_parts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    buffer = []
    end_obj_count = 0
    file_index = 1
    
    for line in cleaned_data:
        buffer.append(line)
        if "End Object" in line:
            end_obj_count += 1
        
        # 修正後的 f-string 語法
        if end_obj_count == chunk_size:
            part_name = os.path.join(output_dir, f"part_{file_index}.txt")
            with open(part_name, 'w', encoding='utf-8') as p_out:
                p_out.write('\n'.join(buffer))
            print(f"    [>] 已生成: {part_name}")
            
            buffer = []
            end_obj_count = 0
            file_index += 1
            
    # 處理最後不足 chunk_size 的剩餘對象 (如最後的 2 個)
    if buffer:
        part_name = os.path.join(output_dir, f"part_{file_index}.txt")
        with open(part_name, 'w', encoding='utf-8') as p_out:
            p_out.write('\n'.join(buffer))
        print(f"    [>] 已生成末尾殘餘塊: {part_name} (對象數: {end_obj_count})")

    print(f"\n[!] 任務達成！")
    print(f"[-] 輸出資料夾: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 確保 blueprint.txt 與此腳本在同一目錄
    process_and_split_blueprint('blueprint.txt', 'blueprint_clean.txt', chunk_size=100)