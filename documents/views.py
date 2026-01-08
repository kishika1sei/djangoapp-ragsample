from django.shortcuts import render,redirect,get_object_or_404
from .models import Document,Department
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseForbidden
from django.views.decorators.http import require_POST
from .forms import DocumentUploadForm
from accounts.services.permissions import can_delete_document
from documents.services.document_service import upload_document
from documents.services.document_service import delete_document as delete_service
from documents.services.document_service import reindex_all_documents
# ログイン済みユーザのみこのメソッドを通す
@login_required
def dashboard(request):
    if request.method == "POST":
        form = DocumentUploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            files = form.cleaned_data["files"]  # ← ここがリストになる

            if request.user.is_superuser:
                dept = form.cleaned_data.get("department") or request.user.department
            else:
                dept = request.user.department

            success, failed = 0, 0
            for f in files:
                try:
                    upload_document(actor=request.user, uploaded_file=f, department=dept)
                    success += 1
                except Exception:
                    failed += 1

            if success and failed == 0:
                messages.success(request, f"ドキュメントをアップロードしました。（{success}件）")
            elif success and failed:
                messages.warning(request, f"一部失敗しました。（成功:{success}件 / 失敗:{failed}件）")
            else:
                messages.error(request, "ドキュメントのアップロードに失敗しました。")

            return redirect("documents:dashboard")
    else:
        form = DocumentUploadForm(user=request.user)

    department_list = Department.objects.all().order_by("id")

    my_dept_list = (
        Document.objects.filter(department=request.user.department)
        .select_related("department", "uploaded_by")
        .order_by("-created_at")
    )

    # 第2セクション用(superuserのみ表示)
    if request.user.is_superuser:
        upload_list = (
            Document.objects.all()
            .select_related("department", "uploaded_by")
            .order_by("-created_at")
        )
    else:
        upload_list = (
            Document.objects.filter(department=request.user.department)
            .select_related("department", "uploaded_by")
            .order_by("-created_at")
        )

    return render(
        request,
        "documents/dashboard.html",
        {
            "upload_list": upload_list,
            "my_dept_list": my_dept_list,
            "department_list": department_list,
            "form": form,
        },
    )


@login_required
@require_POST
def delete_document(request, document_id):
    document = get_object_or_404(Document, id=document_id)

    if not can_delete_document(request.user, document):
        return HttpResponseForbidden()

    try:
        delete_service(actor=request.user, document=document)
        messages.success(request, "ドキュメントを削除しました。")
    except Exception:
        messages.error(request, "ドキュメントの削除に失敗しました。")

    return redirect("documents:dashboard")
     

@login_required
@require_POST
def reindex_all(request):
    if not request.user.is_superuser:
        return HttpResponseForbidden()

    import logging
    logger = logging.getLogger(__name__)

    try:
        result = reindex_all_documents(actor=request.user)
        messages.success(
            request,
            f"全件再インデックス完了（成功: {result['success_documents']} / 失敗: {result['failed_documents']} / 合計: {result['total_documents']}）"
        )
    except Exception:
        logger.exception("reindex_all failed")
        messages.error(request, "全件再インデックスに失敗しました。")

    return redirect("documents:dashboard")