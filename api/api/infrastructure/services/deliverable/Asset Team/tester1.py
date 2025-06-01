"""
Tester file for the RL-based document analysis and study guide generation pipeline.
Provide a markdown file path and run this file to train the model.
"""

from RL_script_Asset import train_document_policy, read_markdown_file, bin_document_concept, bin_document_context, extract_terms_and_citations
import os

# ========== FILL IN YOUR PARAMETERS HERE ==========

# Markdown File Path: Path to your markdown document
# The system will analyze this document to:
# 1. Determine the subject area (concept bin)
# 2. Identify the document type (context bin) 
# 3. Extract key terms and definitions
# 4. Extract important citations and quotes
# 5. Train RL model to optimize study guide generation

markdown_file_path = "sample_biology_textbook.md"  # Replace with your file path

# Optional: Custom log filename
log_filename = "biology_document_training.jsonl"

# ========== SAMPLE DOCUMENT CREATION (for testing) ==========

def create_sample_document():
    """Create a sample biology textbook document for testing"""
    sample_content = """
# Chapter 12: Cell Structure and Function

## Introduction to Cell Biology

The **cell** is the basic unit of life and the smallest structure that can be considered living. All living organisms are composed of one or more cells, making cell biology a fundamental area of study in the life sciences.

### Key Concepts in Cell Biology

**Prokaryotic cells** are cells that lack a membrane-bound nucleus and organelles. The genetic material in prokaryotic cells is freely floating in the cytoplasm. Examples include bacteria and archaea.

**Eukaryotic cells** are cells that contain a membrane-bound nucleus and specialized organelles. The nucleus houses the cell's DNA, which controls cellular activities. Examples include plant cells, animal cells, and fungal cells.

### Cell Theory

The cell theory states three fundamental principles:
1. All living things are made of one or more cells
2. The cell is the basic unit of life
3. All cells come from pre-existing cells

This theory was developed through the work of several scientists including Robert Hooke (1665), who first observed cells in cork tissue under a microscope.

## Cellular Organelles

### The Nucleus
The **nucleus** is often called the "control center" of the cell because it contains the cell's DNA and controls gene expression. The nucleus is surrounded by a double membrane called the nuclear envelope.

### Mitochondria
**Mitochondria** are known as the "powerhouses of the cell" because they produce ATP through cellular respiration. According to recent studies, "a typical human cell contains between 100 and 1,000 mitochondria" (Johnson et al., 2023).

### Ribosomes
**Ribosomes** are small structures responsible for protein synthesis. They can be found free-floating in the cytoplasm or attached to the endoplasmic reticulum.

## Cell Membrane Structure

The **cell membrane** (also called the plasma membrane) is a selectively permeable barrier that surrounds the cell. It consists of a phospholipid bilayer with embedded proteins.

**Phospholipids** are molecules that have a hydrophilic (water-loving) head and hydrophobic (water-fearing) tails. This arrangement allows the cell membrane to control what enters and exits the cell.

## Transport Across Membranes

### Passive Transport
**Passive transport** is the movement of substances across the cell membrane without the use of cellular energy. Examples include:
- **Diffusion**: Movement from high to low concentration
- **Osmosis**: Diffusion of water across a membrane
- **Facilitated diffusion**: Diffusion through protein channels

### Active Transport
**Active transport** requires cellular energy (ATP) to move substances against their concentration gradient. The sodium-potassium pump is a classic example of active transport.

## Important Statistics and Research

According to the National Institute of Health (2023), "the human body contains approximately 37.2 trillion cells." Recent research has shown that cellular dysfunction is linked to numerous diseases, with mitochondrial disorders affecting "1 in 4,000 individuals worldwide" (Cell Biology Research Foundation, 2023).

## Study Questions

1. What are the main differences between prokaryotic and eukaryotic cells?
2. How does the structure of the cell membrane relate to its function?
3. Explain the difference between passive and active transport.

## References

- Johnson, M., Smith, K., & Davis, L. (2023). Mitochondrial distribution in human cells. *Journal of Cell Biology*, 45(3), 123-135.
- National Institute of Health. (2023). Human cellular composition statistics. *NIH Database*.
- Cell Biology Research Foundation. (2023). Global impact of mitochondrial disorders. *Annual Report*.
"""
    
    with open("sample_biology_textbook.md", "w", encoding="utf-8") as f:
        f.write(sample_content)
    print("Sample biology textbook document created: sample_biology_textbook.md")

if __name__ == "__main__":
    print("="*70)
    print("RL-BASED DOCUMENT ANALYSIS & STUDY GUIDE GENERATION")
    print("="*70)
    
    # Create sample document if it doesn't exist (for testing purposes)
    if not os.path.exists(markdown_file_path):
        print(f"File {markdown_file_path} not found. Creating sample document...")
        create_sample_document()
        markdown_file_path = "samplefile.md"

    print(f"Analyzing document: {markdown_file_path}")
    
    # Preview the document analysis
    print("\n" + "="*50)
    print("DOCUMENT PREVIEW & ANALYSIS")
    print("="*50)
    
    # Read the document
    content = read_markdown_file(markdown_file_path)
    if content:
        print(f"Document length: {len(content)} characters")
        print(f"Preview (first 200 chars): {content[:200]}...")
        
        # Show binning results
        print("\nAnalyzing document bins...")
        concept_bin = bin_document_concept(content)
        context_bin = bin_document_context(content)
        print(f"Subject bin: {concept_bin}")
        print(f"Context bin: {context_bin}")
        
        # Show extraction results
        print("\nExtracting terms and citations...")
        terms, citations = extract_terms_and_citations(content)
        
        print(f"\nExtracted {len(terms)} key terms:")
        for i, term in enumerate(terms[:5]):  # Show first 5 terms
            definition_preview = term['definition'][:50] + "..." if len(term['definition']) > 50 else term['definition']
            print(f"  {i+1}. {term['term']}: {definition_preview}")
        if len(terms) > 5:
            print(f"  ... and {len(terms) - 5} more terms")
            
        print(f"\nExtracted {len(citations)} citations:")
        for i, citation in enumerate(citations[:3]):  # Show first 3 citations
            citation_preview = citation[:80] + "..." if len(citation) > 80 else citation
            print(f"  {i+1}. {citation_preview}")
        if len(citations) > 3:
            print(f"  ... and {len(citations) - 3} more citations")
    
    print("\n" + "="*70)
    print("STARTING REINFORCEMENT LEARNING TRAINING")
    print("="*70)
    
    # Train the policy
    policy, logs = train_document_policy(
        markdown_file_path=markdown_file_path,
        filename=log_filename
    )
    
    if policy and logs:
        print("\n" + "="*70)
        print("DOCUMENT ANALYSIS TRAINING COMPLETE")
        print("="*70)
        print("Files created:")
        print(f"- {log_filename} (real-time training logs)")
        print(f"\nDocument successfully processed:")
        print(f"- Subject: {logs[-1].concept_bin}")
        print(f"- Source Type: {logs[-1].context_bin}")
        print(f"- Terms: {logs[-1].terms_count}")
        print(f"- Citations: {logs[-1].citations_count}")
    else:
        print("Error. Check document path.")