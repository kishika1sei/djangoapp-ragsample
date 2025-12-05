from django.shortcuts import render,redirect,get_object_or_404
from .models import Document,Department
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from django.conf import settings
from .forms import DocumentUploadForm
# ログイン済みのみこのメソッドを通す
# system_adminとdept_adminだけがアップロードできるようにする
@login_required
def dashboard(request):
    # POSTリクエストがあったときはアップロードフォームから送信されたファイルをDocumentモデルに保存
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)

        if form.is_valid():
            upload_file = form.cleaned_data['file']
            # 1.ファイルを保存
            save_path = default_storage.save(f'{settings.MEDIA_ROOT}/{upload_file.name}', upload_file)

            # 2.documentインスタンスを作成して保存
            Document.objects.create(
            title=upload_file.name,
            file_path=save_path,
            department=request.user.department,
            uploaded_by=request.user,
            )
            
            return redirect('documents:dashboard')
    # GETリクエストがあった時に既存のドキュメント一覧を表示する
    else:
        #のドキュメント一覧情報を取得して表示する
        form = DocumentUploadForm()

    document_list = Document.objects.filter(department=request.user.department)
    department_list = Department.objects.all()
    # POSTの場合は、ファイルを保存してからドキュメント一覧情報を取得して表示する
    return render(request, 'documents/dashboard.html', {'document_list': document_list, 'department_list': department_list,'form':form})

@login_required
@require_POST
def delete_document(request, document_id):
    try:
        document = get_object_or_404(Document, id=document_id)
        document.delete()
        messages.success(request,"ドキュメントを削除しました。")
    except Exception:
        messages.error(request,"ドキュメントの削除に失敗しました。")
    
    return redirect('documents:dashboard')        

def upload_document(request):
    pass  # アップロード処理は未実装