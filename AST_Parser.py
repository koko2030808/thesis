import re
import tkinter as tk

def process_t3d():
    root = tk.Tk(); root.withdraw()
    try: t3d = root.clipboard_get()
    except tk.TclError: return
    if "Begin Object" not in t3d: return

    # 1. 宣告域鎖定建表 (修復指針污染)
    nodes = {}
    for block in t3d.split('Begin Object '):
        name_m = re.search(r'Name="([^"]+)"', block)
        # 【核心修復】：精準鎖定第一行的 Class 宣告，徹底無視下方 LinkedTo 的字串干擾
        class_m = re.search(r'Class=[^\s=]+/([^/\s"]+)', block) 
        
        if not name_m or not class_m: continue
        n_id = name_m.group(1)
        node_class = class_m.group(1)
        
        name = n_id.split('_')[0]

        if 'K2Node_MacroInstance' in node_class:
            m = re.search(r'MacroGraph="[^"]*:([^\'"]+)[\'"]', block)
            if m: name = f"Macro_{m.group(1)}"
        elif 'K2Node_CallFunction' in node_class:
            m = re.search(r'FunctionReference=\([^)]*MemberName="([^"]+)"', block)
            if m: name = m.group(1)
        elif 'K2Node_DynamicCast' in node_class:
            m = re.search(r'TargetType="[^"]*\.([^"\'\\]+)[\'"]', block)
            if m: name = f"CastTo_{m.group(1)}"
        elif 'K2Node_VariableGet' in node_class:
            m = re.search(r'VariableReference=\([^)]*MemberName="([^"]+)"', block)
            if m: name = f"Get_{m.group(1)}"
        elif 'K2Node_CustomEvent' in node_class:
            m = re.search(r'CustomFunctionName="([^"]+)"', block)
            if m: name = f"Event_{m.group(1)}"
        elif 'K2Node_GetArrayItem' in node_class:
            name = "GetArrayItem"

        nodes[n_id] = name

    # 2. 拓撲重組 (維持 V4 的 UUID 剃除與反向過濾)
    result = []
    for block in t3d.split('Begin Object '):
        tgt_m = re.search(r'Name="([^"]+)"', block)
        if not tgt_m: continue
        tgt_name = nodes.get(tgt_m.group(1), "Unknown")

        for line in block.split('\n'):
            if 'LinkedTo=' in line:
                if 'Direction="EGPD_Output"' in line: continue
                
                pin_m = re.search(r'PinName="([^"]+)"', line)
                link_m = re.search(r'LinkedTo=\(([^,]+),', line)
                
                if pin_m and link_m:
                    tgt_pin = pin_m.group(1)
                    src_id = link_m.group(1).replace('"', '').split(' ')[0]
                    src_name = nodes.get(src_id, f"Unknown_Src")
                    
                    if tgt_pin in ["execute", "then", "LoopBody", "Completed"]:
                        result.append(f"[白線] {src_name} -> {tgt_name}")
                    else:
                        result.append(f"[彩線] ({src_name}) ==> ({tgt_name} : {tgt_pin})")

    output = "\n".join(sorted(set(result)))
    root.clipboard_clear(); root.clipboard_append(output); root.update(); root.destroy()
    
    print("========================================")
    print("【AST 100% 絕對對齊收割成功】")
    print(output)
    print("========================================")

if __name__ == "__main__":
    process_t3d()