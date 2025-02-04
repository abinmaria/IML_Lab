import pdfplumber
import pandas as pd

# Function to convert PDF to CSV
def pdf_to_csv(pdf_file, output_csv_file):
    with pdfplumber.open(pdf_file) as pdf:
        # Extracting all the pages
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text()
        
        # You can also extract tables if you want by using:
        # for page in pdf.pages:
        #     table = page.extract_table()
        #     print(table)  # Example: Print the extracted table
        
        # Assuming the data is in a tabular format, you can convert it to a pandas DataFrame.
        # You can split the text data to rows and columns based on your PDF structure.
        # For example:
        rows = all_text.split('\n')
        data = [row.split() for row in rows]  # Modify this as per how your data is structured

        # Convert data to a DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_csv_file, index=False, header=False)
        print(f"CSV saved as {output_csv_file}")

# Specify the path to your PDF and the desired CSV output filename
pdf_file_path = '/home/abin/IML lab/work.pdf'
output_csv_file = 'converted.csv'

# Call the function
pdf_to_csv(pdf_file_path, output_csv_file)
