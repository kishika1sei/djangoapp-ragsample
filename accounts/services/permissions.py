from accounts.models import User
from documents.models import Document

def can_delete_document(user: User, document: Document) -> bool:
    if not user.is_authenticated:
        return False
    if user.is_superuser:
        return True

    if user.role == User.Role.SYSTEM_ADMIN:
        return True

    if user.role == User.Role.DEPT_ADMIN:
        # 部門未設定の場合は不可に倒す
        if user.department_id is None:
            return False
        return document.department_id == user.department_id

    return False
