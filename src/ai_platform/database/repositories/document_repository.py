from ai_platform.database.connection import get_connection


class DocumentRepository:
    """Provide CRUD operations for documents."""

    def create_document(
        self,
        project_id: int,
        filename: str,
        file_type: str,
        status: str,
    ) -> int:
        """Create a document and return its ID."""
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT INTO documents
                    (project_id, filename, file_type, status)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, filename, file_type, status),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def get_document(
        self,
        document_id: int,
    ) -> dict[str, object] | None:
        """Return a document by ID."""
        connection = get_connection()

        try:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    project_id,
                    filename,
                    file_type,
                    status,
                    created_at
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "document_id": row[0],
                "project_id": row[1],
                "filename": row[2],
                "file_type": row[3],
                "status": row[4],
                "created_at": row[5],
            }
        finally:
            connection.close()

    def get_documents_by_project(
        self,
        project_id: int,
    ) -> list[dict[str, object]]:
        """Return all documents belonging to a project."""
        connection = get_connection()

        try:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    project_id,
                    filename,
                    file_type,
                    status,
                    created_at
                FROM documents
                WHERE project_id = ?
                ORDER BY document_id
                """,
                (project_id,),
            ).fetchall()

            return [
                {
                    "document_id": row[0],
                    "project_id": row[1],
                    "filename": row[2],
                    "file_type": row[3],
                    "status": row[4],
                    "created_at": row[5],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def update_document_status(
        self,
        document_id: int,
        status: str,
    ) -> None:
        """Update a document processing status."""
        connection = get_connection()

        try:
            connection.execute(
                """
                UPDATE documents
                SET status = ?
                WHERE document_id = ?
                """,
                (status, document_id),
            )
            connection.commit()
        finally:
            connection.close()

    def delete_document(self, document_id: int) -> None:
        """Delete a document by ID."""
        connection = get_connection()

        try:
            connection.execute(
                """
                DELETE FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            )
            connection.commit()
        finally:
            connection.close()
