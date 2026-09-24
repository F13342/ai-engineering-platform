from ai_platform.database.repositories.document_repository import (
    DocumentRepository,
)
from ai_platform.database.repositories.project_repository import ProjectRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_document_crud() -> None:
    """Test document CRUD operations and project relationship."""
    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()
    document_repository = DocumentRepository()

    user_id = user_repository.create_user(
        username="RAG User",
        email="rag.user@example.com",
    )

    project_id = project_repository.create_project(
        user_id=user_id,
        name="Knowledge Base",
        description="Project for document-based AI.",
    )

    document_id = document_repository.create_document(
        project_id=project_id,
        filename="machine_learning.pdf",
        file_type="pdf",
        status="uploaded",
    )

    document = document_repository.get_document(document_id)

    assert document is not None
    assert document["project_id"] == project_id
    assert document["filename"] == "machine_learning.pdf"
    assert document["file_type"] == "pdf"
    assert document["status"] == "uploaded"

    documents = document_repository.get_documents_by_project(project_id)

    assert len(documents) == 1
    assert documents[0]["document_id"] == document_id

    document_repository.update_document_status(
        document_id=document_id,
        status="processed",
    )

    updated_document = document_repository.get_document(document_id)

    assert updated_document is not None
    assert updated_document["status"] == "processed"

    document_repository.delete_document(document_id)

    assert document_repository.get_document(document_id) is None

    project_repository.delete_project(project_id)
    user_repository.delete_user(user_id)
