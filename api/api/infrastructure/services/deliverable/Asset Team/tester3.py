"""
Tester file for the RL-based document analysis - HISTORY LECTURE NOTES
This tests the system with history content in a lecture notes format.
"""

import os
from RL_script_Asset import train_document_policy, read_markdown_file, bin_document_concept, bin_document_context, extract_terms_and_citations

# ========== HISTORY LECTURE NOTES TEST ==========

# Markdown File Path: History lecture notes
markdown_file_path = "wwii_causes_lecture.md"

# Custom log filename for this test
log_filename = "history_lecture_training.jsonl"

# ========== SAMPLE HISTORY LECTURE DOCUMENT CREATION ==========

def create_history_lecture_document():
    """Create sample WWII history lecture notes for testing"""
    sample_content = """
# History 102: Modern World History
## Lecture 8: Causes and Origins of World War II
### Professor Johnson | October 15, 2023

---

## Today's Learning Goals
- Understand the multiple causes that led to WWII
- Analyze the failure of the League of Nations
- Examine the rise of totalitarian regimes
- Connect WWI aftermath to WWII origins

---

## I. Introduction: The "Twenty Years' Crisis"

Good morning class. Today we're discussing what historian E.H. Carr called the **"Twenty Years' Crisis"** - the period between WWI and WWII (1919-1939).

### Key Question for Today:
*Was WWII inevitable after WWI, or could it have been prevented?*

---

## II. Long-Term Causes (1919-1933)

### A. Treaty of Versailles and Its Consequences

The **Treaty of Versailles** (1919) created lasting resentment in Germany:

**War Guilt Clause** (Article 231): Germany forced to accept full responsibility for the war. As historian Fritz Stern noted, "The treaty was designed to keep Germany weak, but instead it made Germany angry."

**Reparations**: Germany required to pay massive war reparations. The exact amount was set at 132 billion gold marks in 1921 - approximately $400 billion in today's currency.

**Territorial Losses**: 
- Alsace-Lorraine returned to France
- Saar Basin placed under League of Nations control
- Polish Corridor created, separating East Prussia from Germany
- All German colonies redistributed as League mandates

**Military Restrictions**:
- German army limited to 100,000 men
- No air force permitted
- Navy restricted to 6 battleships, no submarines
- Rhineland to be permanently demilitarized

### B. Economic Instability

**Hyperinflation** in Germany (1921-1923): The German mark became virtually worthless. A loaf of bread that cost 250 marks in January 1923 cost 200 billion marks by November 1923.

**Great Depression** (1929-1939): Global economic collapse that hit Germany particularly hard. "By 1932, over 6 million Germans were unemployed - nearly 30% of the workforce" (Economic History Review, 2019).

---

## III. Rise of Totalitarian Regimes

### A. Fascism in Italy

**Benito Mussolini** and the **Fascist Party** rose to power in 1922:
- Promised to restore Italian greatness and recreate a Roman Empire
- Used violence and intimidation through **Blackshirts** paramilitary groups
- Established the first fascist dictatorship in Europe

**Key Fascist Principles**:
- Extreme nationalism
- Rejection of democracy and liberalism  
- Glorification of violence and war
- Cult of personality around the leader

### B. Nazism in Germany

**Adolf Hitler** and the **Nazi Party** (NSDAP):
- Gained support during economic crisis
- Hitler appointed Chancellor in January 1933
- Quickly dismantled democratic institutions

**Nazi Ideology**:
- **Racial supremacy**: Belief in Aryan racial superiority
- **Lebensraum**: "Living space" - need for German territorial expansion
- **Anti-Semitism**: Systematic persecution of Jewish people
- **Führerprinzip**: Absolute authority of the leader

**Key Events**:
- 1933: Hitler becomes Chancellor, Enabling Act passed
- 1934: Night of the Long Knives, Hitler becomes Führer
- 1935: Nuremberg Laws enacted
- 1938: Kristallnacht ("Night of Broken Glass")

### C. Militarism in Japan

**Japanese Expansionism**:
- Military-dominated government seeking empire in Asia
- Belief in Japanese racial and cultural superiority
- Economic need for raw materials and markets

**Key Aggressive Actions**:
- 1931: Invasion of Manchuria
- 1937: Second Sino-Japanese War begins
- 1940: Tripartite Pact with Germany and Italy

---

## IV. Immediate Causes (1933-1939)

### A. Failure of Collective Security

**League of Nations** proved ineffective:
- No enforcement mechanism for its decisions
- Major powers (US, USSR) not members
- Failed to stop Japanese invasion of Manchuria
- Failed to prevent Italian invasion of Ethiopia (1935)

As Winston Churchill observed in 1938: "The League of Nations has passed from shadow into ghost, from ghost into shadow, and from shadow into nothing."

### B. Policy of Appeasement

**Appeasement**: Policy of making concessions to aggressive powers to avoid war.

**Key Examples**:
- 1936: Germany remilitarizes Rhineland (no response)
- 1938: **Anschluss** - Germany annexes Austria
- 1938: **Munich Agreement** - Sudetenland given to Germany

**Munich Conference** (September 1938):
- British PM Neville Chamberlain declared "peace for our time"
- Czechoslovakia not consulted about loss of Sudetenland
- Demonstrated Western unwillingness to confront Hitler

### C. Breakdown of Diplomacy

**Nazi-Soviet Pact** (August 23, 1939):
- Non-aggression agreement between Hitler and Stalin
- Secret protocol divided Poland between Germany and USSR
- Removed threat of two-front war for Germany

**Polish Guarantee** (March 1939):
- Britain and France promised to defend Poland
- Marked end of appeasement policy
- Set stage for WWII when Germany invaded Poland

---

## V. The Final Crisis: September 1939

### September 1, 1939: Germany Invades Poland
- **Blitzkrieg** ("lightning war") tactics used
- New form of warfare combining tanks, aircraft, and motorized infantry
- Polish forces overwhelmed within weeks

### September 3, 1939: Britain and France Declare War
- Honored their guarantee to Poland
- Marked beginning of WWII in Europe
- Initially a "Phoney War" with little actual fighting

---

## VI. Historical Interpretations

### A. Intentionalist vs. Functionalist Debate

**Intentionalists** argue:
- Hitler had a clear plan for war from the beginning
- WWII was inevitable once Nazis came to power
- War was result of deliberate Nazi policy

**Functionalists** argue:
- War resulted from cumulative radicalization
- Hitler was responding to circumstances, not following master plan
- Multiple factors and actors contributed to outbreak

### B. Role of Economic Factors

Economic historian Adam Tooze argues: "The Nazi economy was structured around preparation for war. By 1939, Germany faced a choice between war and economic collapse."

---

## VII. Conclusion: Lessons for Today

### Key Takeaways:
1. **Multiple Causation**: WWII resulted from complex interaction of factors
2. **Importance of International Cooperation**: Failure of League showed need for effective international institutions
3. **Dangers of Appeasement**: Concessions to aggressors may encourage further aggression
4. **Economic-Political Connection**: Economic instability can fuel political extremism

### Discussion Question:
"Could stronger international institutions have prevented WWII, or were the underlying tensions too great?"

---

## Important Statistics and Data

- "Approximately 70-85 million people died in WWII, making it the deadliest conflict in human history" (World War II Database, 2023)
- "The war cost an estimated $4 trillion in today's dollars" (Economic History Association, 2022)
- "Over 50 countries were involved in the conflict by its end" (International Relations Quarterly, 2023)

---

## Required Readings for Next Class:
- Kershaw, Ian. *To Hell and Back: Europe 1914-1949*, Chapters 8-9
- Primary source: Churchill's "Iron Curtain" speech (1946)
- Document analysis: Excerpts from Nuremberg Trial transcripts

---

## Key Terms Review:

**Totalitarianism**: A system of government that has absolute control over all aspects of public and private life.

**Appeasement**: The policy of making concessions to an aggressive power in order to avoid conflict.

**Blitzkrieg**: A military tactic designed to create psychological shock through rapid, concentrated attacks.

**Anschluss**: The annexation of Austria into Nazi Germany in 1938.

**Lebensraum**: Nazi concept of acquiring territory for German expansion and settlement.

**Fascism**: A far-right, authoritarian political ideology characterized by dictatorial power and extreme nationalism.

---

*Next Lecture: "The Course of WWII: Major Battles and Turning Points"*

**Assignment Due**: 2-page analysis of primary sources from Munich Conference - due next Friday.
"""
    
    with open("wwii_causes_lecture.md", "w", encoding="utf-8") as f:
        f.write(sample_content)
    print("Sample WWII history lecture notes created: wwii_causes_lecture.md")

# ========== RUN THE ANALYSIS ==========

if __name__ == "__main__":
    print("="*75)
    print("RL-BASED DOCUMENT ANALYSIS - HISTORY LECTURE NOTES")
    print("="*75)
    
    # Create sample document if it doesn't exist
    if not os.path.exists(markdown_file_path):
        print(f"File {markdown_file_path} not found. Creating sample document...")
        create_history_lecture_document()
    
    print(f"Analyzing document: {markdown_file_path}")
    
    # Preview the document analysis
    print("\n" + "="*60)
    print("HISTORY LECTURE DOCUMENT PREVIEW & ANALYSIS")
    print("="*60)
    
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
        print(f"Expected: World History subject + Lecture notes context")
        
        # Show extraction results
        print("\nExtracting terms and citations...")
        terms, citations = extract_terms_and_citations(content)
        
        print(f"\nExtracted {len(terms)} historical terms:")
        for i, term in enumerate(terms[:5]):  # Show first 5 terms
            definition_preview = term['definition'][:60] + "..." if len(term['definition']) > 60 else term['definition']
            print(f"  {i+1}. {term['term']}: {definition_preview}")
        if len(terms) > 5:
            print(f"  ... and {len(terms) - 5} more terms")
            
        print(f"\nExtracted {len(citations)} historical citations/quotes:")
        for i, citation in enumerate(citations[:3]):  # Show first 3 citations
            citation_preview = citation[:85] + "..." if len(citation) > 85 else citation
            print(f"  {i+1}. {citation_preview}")
        if len(citations) > 3:
            print(f"  ... and {len(citations) - 3} more citations")
    
    print("\n" + "="*75)
    print("STARTING HISTORY LECTURE REINFORCEMENT LEARNING TRAINING")
    print("="*75)
    
    # Train the policy
    policy, logs = train_document_policy(
        markdown_file_path=markdown_file_path,
        filename=log_filename
    )
    
    if policy and logs:
        print("\n" + "="*75)
        print("HISTORY LECTURE TRAINING COMPLETE")
        print("="*75)
        print("Files created:")
        print(f"- {log_filename} (real-time training logs)")
        print(f"\nHistory lecture document successfully processed:")
        print(f"- Subject: {logs[-1].concept_bin}")
        print(f"- Source Type: {logs[-1].context_bin}")
        print(f"- Terms: {logs[-1].terms_count}")
        print(f"- Citations: {logs[-1].citations_count}")
    else:
        print("Training failed! Check the document path and try again.")