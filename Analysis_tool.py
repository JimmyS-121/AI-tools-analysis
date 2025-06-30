import pandas as pd
import re
from typing import Dict

class DataCleaner:
    def __init__(self):
        # Target column mapping (exact mapping to desired output)
        self.target_columns = {
            'timestamp': ['timestamp', 'date', 'time', 'datetime'],
            'department': ['department', 'dept', 'which department'],
            'job_role': ['job_role', 'role', 'position'],
            'ai_tool': ['ai_tool', 'what ai tool', 'ai tool used'],
            'usage_frequency': ['usage_frequency', 'usage of ai tools'],
            'purpose': ['purpose', 'purpose of using ai tools'],
            'ease_of_use': ['ease_of_use', 'ease of use'],
            'efficiency': ['efficiency', 'how efficiency'],
            'suggestions': ['suggestions', 'any suggestions']
        }
        
        # Standardization rules for values
        self.standardization_rules = {
            'ai_tool': {
                'chatgpt': 'ChatGPT',
                'poe': 'Poe',
                'canava': 'Canva',
                'gamma': 'Gamma',
                'midjourney': 'Midjourney',
                'copilot': 'Copilot',
                'kling ai': 'Kling AI',
                'deepseek': 'Deepseek'
            },
            'usage_frequency': {
                'weekly': 'Weekly',
                'monthly': 'Monthly',
                'daily': 'Daily',
                'rarely': 'Rarely'
            }
        }

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("Empty file uploaded")
            return df
        except Exception as e:
            raise ValueError(f"Error loading file: {str(e)}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataframe"""
        if df.empty:
            raise ValueError("Empty dataframe provided")
        
        # Step 1: Standardize column names
        df = self._standardize_column_names(df)
        
        # Step 2: Clean each column
        for col in df.columns:
            if col in self.standardization_rules:
                df[col] = self._standardize_values(df[col], col)
        
        # Step 3: Ensure proper data types
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        if 'ease_of_use' in df.columns:
            df['ease_of_use'] = pd.to_numeric(df['ease_of_use'], errors='coerce')
        
        if 'efficiency' in df.columns:
            df['efficiency'] = pd.to_numeric(df['efficiency'], errors='coerce')
        
        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert input columns to target column names"""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            matched = False
            
            # Check for exact matches first
            for target_col, variations in self.target_columns.items():
                if col_lower in [v.lower() for v in variations]:
                    column_mapping[col] = target_col
                    matched = True
                    break
            
            # If no exact match, check for partial matches
            if not matched:
                for target_col, variations in self.target_columns.items():
                    for variation in variations:
                        if variation.lower() in col_lower:
                            column_mapping[col] = target_col
                            matched = True
                            break
                    if matched:
                        break
            
            # If still no match, keep original column
            if not matched:
                column_mapping[col] = col
        
        return df.rename(columns=column_mapping)

    def _standardize_values(self, series: pd.Series, column: str) -> pd.Series:
        """Standardize values in a column"""
        if column not in self.standardization_rules:
            return series
        
        # Convert to string and clean
        cleaned = series.astype(str).str.strip().str.lower()
        
        # Apply standardization rules
        rules = self.standardization_rules[column]
        for original, standardized in rules.items():
            cleaned = cleaned.str.replace(original, standardized, regex=False)
        
        return cleaned.str.title()

    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        """Save cleaned data to CSV"""
        cleaned_df = self.clean_data(df)
        cleaned_df.to_csv(output_path, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    cleaner = DataCleaner()
    try:
        # Example usage:
        input_file = "Testing use (Responses) - Form responses 1.csv"
        output_file = "cleaned_questionnaire.csv"
        
        raw_data = cleaner.load_data(input_file)
        cleaner.save_cleaned_data(raw_data, output_file)
        print(f"Success! Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
