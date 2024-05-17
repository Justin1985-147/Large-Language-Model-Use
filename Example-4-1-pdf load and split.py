# RAG preparation: An example of document loading and splitting
"""
The main function of this code is to extract text from a PDF file and organize it into paragraphs.
It first traverses each page of the PDF file, extracts the text of each page, and then splits the text into paragraphs by blank lines.
Finally, it prints out the first three paragraphs.
"""
#!pip install --upgrade openai
# Install pdf parsing library
#!pip install pdfminer.six

# Import relevant modules from the pdfminer library for extracting text from PDF files
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# Define a function to extract text from a PDF file
def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''Extract text from a PDF file (by specified page number)'''
    paragraphs = []  # Create an empty list to store the extracted paragraphs
    buffer = ''  # Create an empty string to temporarily store text
    full_text = ''  # Create an empty string to store the complete text
    # Traverse each page of the PDF file
    for i, page_layout in enumerate(extract_pages(filename)):
        # If a page range is specified, skip pages outside the range
        if page_numbers is not None and i not in page_numbers:
            continue
        # Traverse each element of the page
        for element in page_layout:
            # If the element is a text container, extract its text
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # Separate the text into paragraphs by blank lines
    lines = full_text.split('\n')
    # Traverse each line of text
    for text in lines:
        # If the length of the text is greater than or equal to the minimum line length
        if len(text) >= min_line_length:
            # If the text does not end with a hyphen, add it to the buffer and add a space in front
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        # If the length of the text is less than the minimum line length and the buffer is not empty
        elif buffer:
            # Add the content of the buffer to the paragraph list and clear the buffer
            paragraphs.append(buffer)
            buffer = ''
    # If the buffer is not empty, add its content to the paragraph list
    if buffer:
        paragraphs.append(buffer)
    # Return the paragraph list
    return paragraphs

# Extract text from the PDF file, with a minimum line length of 10
paragraphs = extract_text_from_pdf("llama2.pdf", min_line_length=10)

# Print the first three paragraphs
for para in paragraphs[:3]:
    print(para+"\n")