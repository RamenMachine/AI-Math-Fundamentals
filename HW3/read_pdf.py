import PyPDF2

with open('python-week3(1).pdf', 'rb') as pdfFile:
    pdfReader = PyPDF2.PdfReader(pdfFile)
    
    for pageNum in range(len(pdfReader.pages)):
        page = pdfReader.pages[pageNum]
        print(f"\n=== PAGE {pageNum + 1} ===\n")
        print(page.extract_text())
