import os

def generate_clean_tree(startpath, exclude_dirs):
    for root, dirs, files in os.walk(startpath):
        # Can thiệp vào list dirs để os.walk bỏ qua các thư mục này
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Tính toán mức độ thụt lề (indentation)
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level
        
        # In tên thư mục
        folder_name = os.path.basename(root) if root != startpath else os.path.basename(os.path.abspath(startpath))
        print(f"{indent}├── {folder_name}/")
        
        # In tên các tệp tin bên trong
        subindent = '│   ' * (level + 1)
        for f in files:
            # Bỏ qua các tệp ẩn hoặc tệp cấu hình không quan trọng nếu cần
            if not f.endswith(('.pyc', '.bag', '.db3')): 
                print(f"{subindent}├── {f}")

if __name__ == "__main__":
    # Danh sách các thư mục rác/cấu hình cần loại bỏ khỏi cây
    ignore_list = {'.venv', 'venv', '__pycache__', '.git', '.vscode', '.idea'}
    
    print("\n--- CẤU TRÚC THƯ MỤC HIỆN TẠI ---")
    generate_clean_tree('.', ignore_list)
    print("----------------------------------\n")