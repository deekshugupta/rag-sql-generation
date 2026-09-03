
import pymupdf as fitz  # PyMuPDF
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()  

PDF_FILE_PATH = "pdf/travel_rag.pdf" # Currently using from local but can be changed to S3 



class ColumnSchema(BaseModel):
    name: str = Field(description="Column name")
    data_type: str = Field(description="Data type, e.g., INT, VARCHAR(100), DATETIME")
    is_primary_key: bool = Field(description="Set TRUE if column is PK or part of primary key")
    is_foreign_key: bool = Field(description="Set TRUE if column references another table (FK)")
    foreign_key_target: str = Field(
        default="", 
        description="Target table and column if FK (e.g., 'users.user_id'). Empty string if not FK."
    )
    nullable: Literal["YES", "NO"] = Field(description="NO if NOT NULL, YES if nullable")
    business_meaning: str = Field(description="Description or business meaning of the column")

class Relationship(BaseModel):
    source_column: str = Field(description="Local table joining column (e.g., user_id)")
    target_table: str = Field(description="Name of the referenced foreign table")
    target_column: str = Field(description="Target joining column (e.g., user_id)")
    cardinality: str = Field(description="Cardinality: '1:N', 'N:1', '1:1'")
    is_optional: bool = Field(description="TRUE if join uses optional/nullable key (e.g. LEFT JOIN required)")
    join_condition: str = Field(description="SQL join condition e.g. u.user_id = b.user_id")
    notes: Optional[str] = Field(description="Usage rules or LEFT JOIN recommendations")


class TableSchema(BaseModel):
    database_name: str = Field(description="Database name (e.g. travel_rag)")
    table_name: str = Field(description="Name of the table")
    alias: str = Field(description="Recommended single/short letter table alias")
    business_role: str = Field(description="Business role or purpose of this table")
    synonyms: List[str] = Field(description="Synonyms, alternative business terms, or query vocabulary")
    columns: List[ColumnSchema] = Field(description="List of all table columns")
    relationships: List[Relationship] = Field(description="List of table relationships and join paths")
    sql_rules: List[str] = Field(description="SQL constraints, aggregation rules, and filtering guidelines")


class DatabaseSchemaExtraction(BaseModel):
    tables: List[TableSchema] = Field(description="List of all extracted tables from schema")




class DynamicSchemaChunker:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        ).with_structured_output(DatabaseSchemaExtraction)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts raw text automatically from input PDF file."""
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        return full_text

    def parse_schema_with_llm(self, raw_text: str) -> DatabaseSchemaExtraction:
        """Dynamically parses unstructured schema text into structured schema entities."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Data Architect. Extract all database schema metadata, "
                       "tables, column definitions, relationships, synonyms, and business rules "
                       "from the provided documentation into structured format."),
            ("human", "Schema Documentation:\n{raw_text}")
        ])
        chain = prompt | self.llm
        return chain.invoke({"raw_text": raw_text})

    def generate_rag_chunks(self, schema_data: DatabaseSchemaExtraction) -> List[dict]:
        """Automatically constructs individual entity-centric Markdown chunks for vector embedding."""
        chunks = []
        for table in schema_data.tables:
            md = [
                f"# Database: {table.database_name}",
                f"## Table: {table.table_name}",
                f"**Business Role:** {table.business_role}",
                f"**Recommended Alias:** `{table.alias}`",
                f"**Synonyms / Keywords:** {', '.join(table.synonyms)}",
                "",
                "### Column Schema",
                "| Column Name | Data Type | Key | Nullable | Meaning |",
                "|---|---|---|---|---|"
            ]
            
            for col in table.columns:
                md.append(f"| {col.name} | {col.data_type} | {col.nullable} | {col.business_meaning} |")
                
            md.append("\n### Relationships & Foreign Keys")
            for rel in table.relationships:
                md.append(f"- **JOIN {rel.target_table}:** `{rel.join_condition}` ({rel.cardinality}) - {rel.notes or ''}")
                
            md.append("\n### SQL Generation Rules")
            for rule in table.sql_rules:
                md.append(f"- {rule}")

            chunk_text = "\n".join(md)
            
            chunks.append({
                "id": f"{table.database_name}.{table.table_name}",
                "text": chunk_text,
                "metadata": {
                    "database": table.database_name,
                    "table_name": table.table_name,
                    "columns": [c.name for c in table.columns],
                    "synonyms": table.synonyms
                }
            })
        return chunks

    def process_pdf(self) -> List[dict]:
        """End-to-End Automated Pipeline."""
        raw_text = self.extract_text_from_pdf(PDF_FILE_PATH)
        structured_schema = self.parse_schema_with_llm(raw_text)
        return self.generate_rag_chunks(structured_schema)

# ==========================================
# 3. Execution Example
# ==========================================
if __name__ == "__main__":

    chunker = DynamicSchemaChunker()
    
    # Process PDF automatically without any manual schema declaration
    auto_chunks = chunker.process_pdf()

    print(f"Successfully processed PDF. Generated {len(auto_chunks)} vector-ready table chunks.\n")
    print("--- GENERATED CHUNK ---")
    print(auto_chunks[0]["text"])

