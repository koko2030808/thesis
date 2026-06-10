import cv2

def visual_frame_finder(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 無法打開影片，請檢查路徑。")
        return

    frame_idx = 0
    print("\n>>> 啟動視覺鎖定工具...")
    print("操作說明: [D] 下一幀 | [A] 上一幀 | [W] 跳 50 幀 | [S] 退 50 幀 | [Q] 退出並記錄數字")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break

        # 在畫面上渲染幀索引
        text = f"Frame Index: {frame_idx}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        cv2.imshow("V167 Visual Alignment HUD", frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'): frame_idx += 1
        elif key == ord('a'): frame_idx = max(0, frame_idx - 1)
        elif key == ord('w'): frame_idx += 50
        elif key == ord('s'): frame_idx = max(0, frame_idx - 50)
        elif key == ord('q'): 
            print(f"\n✅ 語義鎖定成功！最終 Sitting Index: {frame_idx}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 請放入你的 RGB 影片路徑 (例如 S9 Sitting 1 的影片)
    video_path = r'D:\videopose2\VideoPose3D\S9\Sitting.60457274.mp4'
    visual_frame_finder(video_path)