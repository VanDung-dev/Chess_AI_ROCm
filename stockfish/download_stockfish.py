import os
import requests
import tarfile
import shutil
from tqdm import tqdm

def download_stockfish():
    """
    Tải về Stockfish từ GitHub và trích xuất nó vào thư mục hiện tại.
    Chỉ giữ lại file binary, xóa toàn bộ file rác và file nén sau khi hoàn tất.
    """

    # Tên file binary và file nén
    binary_name = 'stockfish-ubuntu-x86-64-avx2'
    archive_name = 'stockfish.tar'
    model_url = 'https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar'

    # Đường dẫn trong project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    binary_path = os.path.join(script_dir, binary_name)
    archive_path = os.path.join(script_dir, archive_name)
    inner_folder = os.path.join(script_dir, 'stockfish')
    extracted_binary = os.path.join(inner_folder, binary_name)

    # Kiểm tra nếu binary đã tồn tại
    if os.path.isfile(binary_path):
        print('File stockfish đã tồn tại. Bỏ qua quá trình tải và giải nén.')
        return

    # Nếu chưa có file nén thì bắt đầu tải
    if not os.path.exists(archive_path):
        print('Đang tải xuống...')
        response = requests.get(model_url, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024

            with open(archive_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in response.iter_content(block_size):
                        f.write(chunk)
                        pbar.update(len(chunk))

            print('Tải xuống hoàn tất.')

            # Giải nén file tar
            print('Đang giải nén file TAR...')
            with tarfile.open(archive_path) as tar_ref:
                tar_ref.extractall(script_dir)
            print('Giải nén hoàn tất.')

            # Kiểm tra file stockfish có tồn tại không
            if not os.path.exists(extracted_binary):
                print(f'Lỗi: Không tìm thấy file "{os.path.basename(extracted_binary)}" sau khi giải nén.')
                return

            # Di chuyển file binary ra ngoài thư mục gốc
            print(f'Đổi tên file thành {binary_name}...')
            os.rename(extracted_binary, binary_path)

            # Xóa thư mục con chứa file rác
            if os.path.exists(inner_folder):
                shutil.rmtree(inner_folder)

            # Xóa file nén
            if os.path.exists(archive_path):
                os.remove(archive_path)

            print('Hoàn tất!')

        else:
            print(f'Tải file thất bại. Mã lỗi HTTP: {response.status_code}')
    else:
        print(f'File nén đã tồn tại tại {archive_path}. Bỏ qua tải xuống.')

