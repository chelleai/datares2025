"""
Tester file for the RL-based document analysis - PHYSICS LAB MANUAL
This tests the system with physics content in a lab manual format.
"""

import os
from RL_script_Asset import train_document_policy, read_markdown_file, bin_document_concept, bin_document_context, extract_terms_and_citations

# ========== PHYSICS LAB MANUAL TEST ==========

# Markdown File Path: Physics lab manual
markdown_file_path = "physics_mechanics_lab.md"

# Custom log filename for this test
log_filename = "physics_lab_training.jsonl"

# ========== SAMPLE PHYSICS LAB DOCUMENT CREATION ==========

def create_physics_lab_document():
    sample_content = """
# Physics 101 Laboratory Manual
## Lab 3: Newton's Laws of Motion and Friction

### Learning Objectives
By the end of this lab, students will be able to:
- Apply Newton's second law to analyze motion
- Calculate coefficients of friction
- Understand the relationship between force, mass, and acceleration

### Pre-Lab Reading

#### Newton's Laws of Motion

**Newton's First Law** (Law of Inertia) states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.

**Newton's Second Law** states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. This is expressed mathematically as F = ma, where F is force (measured in Newtons), m is mass (measured in kilograms), and a is acceleration (measured in meters per second squared).

**Newton's Third Law** states that for every action, there is an equal and opposite reaction.

#### Friction Forces

**Static friction** is the force that prevents an object from moving when a force is applied to it. The maximum static friction force is given by f_s,max = μ_s × N, where μ_s is the coefficient of static friction and N is the normal force.

**Kinetic friction** (also called sliding friction) is the force that opposes the motion of an object sliding across a surface. The kinetic friction force is given by f_k = μ_k × N, where μ_k is the coefficient of kinetic friction.

**Coefficient of friction** is a dimensionless number that represents the ratio of the friction force between two bodies and the force pressing them together.

### Experimental Setup

#### Materials Required
- Wooden block with hook
- Set of masses (100g, 200g, 500g)
- Spring scale (0-10 N)
- Inclined plane (adjustable angle)
- Protractor
- Stopwatch
- Meter stick

#### Safety Considerations
"Always ensure the inclined plane is securely positioned before beginning measurements" (Physics Department Safety Manual, 2023). Students must wear safety glasses when handling the spring scale under tension.

### Procedure

#### Part A: Measuring Static Friction

1. Place the wooden block on the horizontal surface
2. Attach the spring scale to the hook on the block
3. Slowly increase the applied force until the block just begins to move
4. Record the maximum force reading as F_static
5. Repeat for different normal forces by adding masses

**Data Collection Note**: "The coefficient of static friction typically ranges from 0.1 to 1.5 for most material combinations" (Halliday, Resnick, & Walker, 2021).

#### Part B: Measuring Kinetic Friction

1. With the block in motion, record the force required to maintain constant velocity
2. This force equals the kinetic friction force F_kinetic
3. Calculate μ_k = F_kinetic / N for each trial

#### Part C: Inclined Plane Analysis

1. Set the inclined plane to various angles (15°, 30°, 45°)
2. Determine the angle at which the block just begins to slide
3. At this **critical angle** θ_c, the coefficient of static friction equals tan(θ_c)

### Data Analysis

#### Sample Calculations

For a 500g block on a horizontal surface:
- Normal force: N = mg = (0.5 kg)(9.8 m/s²) = 4.9 N
- If maximum static friction = 2.45 N, then μ_s = 2.45 N / 4.9 N = 0.5

#### Error Analysis

**Systematic errors** may arise from:
- Calibration issues with the spring scale
- Non-uniform surface roughness
- Air resistance (negligible at low speeds)

**Random errors** include:
- Variations in applied force
- Measurement uncertainty in angle readings
- Timing variations in motion detection

### Results and Discussion

According to recent research, "the coefficient of kinetic friction is typically 10-30% lower than the coefficient of static friction for the same material pair" (Journal of Applied Physics, 2023).

#### Expected Results
- Coefficient of static friction for wood on wood: μ_s ≈ 0.5-0.7
- Coefficient of kinetic friction for wood on wood: μ_k ≈ 0.3-0.5
- Critical angle for wooden block: θ_c ≈ 27-35°

### Post-Lab Questions

1. **Force Analysis**: If a 2 kg block requires 8 N to overcome static friction, what is the coefficient of static friction?

2. **Inclined Plane**: At what angle will a block with μ_s = 0.6 begin to slide down an inclined plane?

3. **Real-World Application**: How do these principles apply to car braking systems and tire design?

### Important Formulas Summary

- **Newton's Second Law**: F_net = ma
- **Static Friction**: f_s ≤ μ_s × N
- **Kinetic Friction**: f_k = μ_k × N
- **Normal Force on Incline**: N = mg cos(θ)
- **Component Force on Incline**: F_parallel = mg sin(θ)

### References and Citations

- Halliday, D., Resnick, R., & Walker, J. (2021). *Fundamentals of Physics*. 12th Edition, Wiley.
- Physics Department Safety Manual. (2023). University Laboratory Guidelines.
- Journal of Applied Physics. (2023). "Modern friction coefficient measurements in educational settings." Vol. 134, Issue 12.
- NASA Technical Publication. (2022). "Friction coefficients for aerospace applications." NASA-TP-2022-220456.

### Additional Notes

The **work-energy theorem** states that the work done on an object equals its change in kinetic energy: W = ΔKE = ½mv² - ½mv₀².

For **uniform circular motion**, the centripetal force is given by F_c = mv²/r, where v is the tangential velocity and r is the radius of the circular path.

Research shows that "friction coefficients can vary by up to 20% depending on environmental conditions such as humidity and temperature" (Materials Science Quarterly, 2023).
"""
    
    with open("physics_mechanics_lab.md", "w", encoding="utf-8") as f:
        f.write(sample_content)
    print("Sample physics lab manual document created: physics_mechanics_lab.md")

# ========== RUN THE ANALYSIS ==========

if __name__ == "__main__":
    print("="*75)
    print("RL-BASED DOCUMENT ANALYSIS - PHYSICS LAB MANUAL")
    print("="*75)
    
    # Create sample document if it doesn't exist
    if not os.path.exists(markdown_file_path):
        print(f"File {markdown_file_path} not found. Creating sample document...")
        create_physics_lab_document()
    
    print(f"Analyzing document: {markdown_file_path}")
    
    # Preview the document analysis
    print("\n" + "="*55)
    print("PHYSICS LAB DOCUMENT PREVIEW & ANALYSIS")
    print("="*55)
    
    # Read and analyze the document
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
        print(f"Expected: Physics subject + Lab manual context")
        
        # Show extraction results
        print("\nExtracting terms and citations...")
        terms, citations = extract_terms_and_citations(content)
        
        print(f"\nExtracted {len(terms)} physics terms:")
        for i, term in enumerate(terms[:5]):  # Show first 5 terms
            definition_preview = term['definition'][:60] + "..." if len(term['definition']) > 60 else term['definition']
            print(f"  {i+1}. {term['term']}: {definition_preview}")
        if len(terms) > 5:
            print(f"  ... and {len(terms) - 5} more terms")
            
        print(f"\nExtracted {len(citations)} citations/quotes:")
        for i, citation in enumerate(citations[:3]):  # Show first 3 citations
            citation_preview = citation[:85] + "..." if len(citation) > 85 else citation
            print(f"  {i+1}. {citation_preview}")
        if len(citations) > 3:
            print(f"  ... and {len(citations) - 3} more citations")
    
    print("\n" + "="*75)
    print("STARTING PHYSICS LAB REINFORCEMENT LEARNING TRAINING")
    print("="*75)
    
    # Train the policy
    policy, logs = train_document_policy(
        markdown_file_path=markdown_file_path,
        filename=log_filename
    )
    
    if policy and logs:
        print("\n" + "="*75)
        print("PHYSICS LAB TRAINING COMPLETE")
        print("="*75)
        print("Files created:")
        print(f"- {log_filename} (real-time training logs)")
        print(f"\nPhysics lab document successfully processed:")
        print(f"- Subject: {logs[-1].concept_bin}")
        print(f"- Source Type: {logs[-1].context_bin}")
        print(f"- Terms: {logs[-1].terms_count}")
        print(f"- Citations: {logs[-1].citations_count}")
    else:
        print("Training failed! Check the document path and try again.")