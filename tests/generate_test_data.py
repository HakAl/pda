import os
import stat
import shutil
import pikepdf
from docx import Document

# --- Configuration ---
# Create a dedicated folder for bad documents inside the 'tests' directory.
# This keeps test assets separate from actual application data.
TEST_DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "bad_documents")


# --- Generator Functions ---

def create_corrupted_files(folder=TEST_DOCS_FOLDER):
    """Creates files with valid extensions but invalid content."""
    # Corrupted PDF
    pdf_path = os.path.join(folder, "corrupted_document.pdf")
    with open(pdf_path, "w") as f:
        f.write("This is not a real PDF file and should fail parsing.")
    print(f"Created: {pdf_path}")

    # Corrupted DOCX
    docx_path = os.path.join(folder, "corrupted_word_doc.docx")
    with open(docx_path, "w") as f:
        f.write("This is not a valid docx zip archive.")
    print(f"Created: {docx_path}")


def create_password_protected_pdf(folder=TEST_DOCS_FOLDER):
    """Creates a simple, password-protected PDF."""
    file_path = os.path.join(folder, "protected.pdf")
    pdf = pikepdf.Pdf.new()
    # No need to add a page for this test case
    pdf.save(file_path, encryption=pikepdf.Encryption(owner="owner", user="user_pass", R=4))
    print(f"Created: {file_path}")


def create_empty_word_doc(folder=TEST_DOCS_FOLDER):
    """Creates a Word document with no readable content."""
    file_path = os.path.join(folder, "empty.docx")
    doc = Document()
    doc.save(file_path)
    print(f"Created: {file_path}")


def create_bad_encoding_txt(folder=TEST_DOCS_FOLDER):
    """Creates a text file with an encoding the loader won't expect (UTF-16)."""
    file_path = os.path.join(folder, "bad_encoding.txt")
    with open(file_path, "w", encoding="utf-16") as f:
        f.write("This file uses UTF-16 encoding and will cause a UnicodeDecodeError.")
    print(f"Created: {file_path}")


def create_permission_denied_file(folder=TEST_DOCS_FOLDER):
    """Creates a file and then removes read permissions to trigger a PermissionError."""
    file_path = os.path.join(folder, "permission_denied.txt")
    with open(file_path, "w") as f:
        f.write("You should not be able to read this.")

    # Set permissions to write-only to allow cleanup, but block reading.
    os.chmod(file_path, stat.S_IWUSR)
    print(f"Created: {file_path} (read access denied)")


# --- Main Execution Block ---

def main():
    """Main function to clean up and generate all test files."""
    print("-" * 50)
    print(f"Preparing to generate test documents in: {TEST_DOCS_FOLDER}")

    if os.path.exists(TEST_DOCS_FOLDER):
        print("Cleaning up old test files...")
        shutil.rmtree(TEST_DOCS_FOLDER)

    os.makedirs(TEST_DOCS_FOLDER)
    print("Created test directory.")

    print("Generating new test files...")
    create_corrupted_files()
    create_password_protected_pdf()
    create_empty_word_doc()
    create_bad_encoding_txt()
    create_permission_denied_file()

    print("\nGeneration complete.")
    print("-" * 50)


if __name__ == "__main__":
    main()