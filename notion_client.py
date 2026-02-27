"""Notion API client for managing investment updates and database entries."""
import os
import re
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime
import config

NOTION_VERSION = "2022-06-28"
NOTION_API = "https://api.notion.com/v1"


class NotionClient:
    """Client for interacting with Notion API."""
    
    def __init__(self):
        self.api_key = os.getenv("NOTION_API_KEY") or os.getenv("NOTION_KEY")
        if not self.api_key:
            raise ValueError("Set NOTION_API_KEY in environment or .env file")
        self.parent_db_id = (os.getenv("NOTION_PARENT_PAGE_ID") or "2ff3ea66f70280e89c3ddde8a4dc3694").replace("-", "")
    
    def _headers(self) -> Dict[str, str]:
        """Return headers for Notion API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION,
        }
    
    def _rich_text_chunks(self, text: str, max_chars: int = 2000) -> List[Dict]:
        """Split text into Notion-compatible rich_text chunks (max 2000 chars each)."""
        out = []
        text = (text or "").replace("\x00", "")
        while text:
            chunk = text[:max_chars]
            if len(text) > max_chars:
                # Try to break at last newline
                last_nl = chunk.rfind("\n")
                if last_nl > max_chars // 2:
                    chunk = chunk[: last_nl + 1]
                    text = text[last_nl + 1 :]
                else:
                    text = text[max_chars:]
            else:
                text = ""
            out.append({"type": "text", "text": {"content": chunk, "link": None}})
        return out
    
    def find_or_create_investment(
        self,
        investment_name: str,
        asset_class: str,
        vintage_year: Optional[str],
        trend: Optional[str] = None,
    ) -> str:
        """
        Find or create an investment entry in the Notion database.
        Returns the investment's page ID.
        """
        # First, try to find existing investment
        existing = self._find_investment_by_name(investment_name)
        if existing:
            print(f"  Found existing investment: {investment_name} (ID: {existing})")
            # Update trend if provided
            if trend:
                self._update_investment_trend(existing, trend)
            return existing
        
        # Create new investment entry
        print(f"  Creating new investment: {investment_name}")
        return self._create_investment_entry(
            investment_name, asset_class, vintage_year, trend
        )
    
    def _find_investment_by_name(self, investment_name: str) -> Optional[str]:
        """Search for an investment by name in the database. Returns page ID if found."""
        try:
            # Query the database for matching entries
            response = requests.post(
                f"{NOTION_API}/databases/{self.parent_db_id}/query",
                headers=self._headers(),
                json={
                    "filter": {
                        "property": "Investment Name",
                        "title": {
                            "equals": investment_name
                        }
                    }
                },
                timeout=15,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if results:
                return results[0]["id"]
            return None
        except Exception as e:
            print(f"    Error searching for investment: {e}")
            return None
    
    def _create_investment_entry(
        self,
        name: str,
        asset_class: str,
        vintage_year: Optional[str],
        trend: Optional[str],
    ) -> str:
        """Create a new investment entry in the Notion database."""
        try:
            payload = {
                "parent": {"database_id": self.parent_db_id},
                "properties": {
                    "Investment Name": {
                        "title": [{"type": "text", "text": {"content": name[:2000], "link": None}}]
                    }
                }
            }
            
            # Add optional properties if the database has them
            if asset_class:
                payload["properties"]["Asset Class"] = {
                    "select": {"name": asset_class}
                }
            
            if vintage_year:
                payload["properties"]["Vintage"] = {
                    "select": {"name": vintage_year}
                }
            
            if trend:
                payload["properties"]["Trend"] = {
                    "select": {"name": trend}
                }
            
            response = requests.post(
                f"{NOTION_API}/pages",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["id"]
        except requests.HTTPError as e:
            print(f"    HTTP Error creating investment: {e.response.status_code} {e.response.text[:200]}")
            raise
        except Exception as e:
            print(f"    Error creating investment: {e}")
            raise
    
    def _update_investment_trend(self, investment_id: str, trend: str) -> None:
        """Update the Trend field of an investment entry."""
        try:
            response = requests.patch(
                f"{NOTION_API}/pages/{investment_id}",
                headers=self._headers(),
                json={
                    "properties": {
                        "Trend": {
                            "select": {"name": trend}
                        }
                    }
                },
                timeout=15,
            )
            response.raise_for_status()
            print(f"  Updated trend to: {trend}")
        except Exception as e:
            print(f"  Warning: Failed to update trend: {e}")
    
    def create_child_page_for_update(
        self,
        parent_page_id: str,
        update_date: str,
        update_content: str,
    ) -> str:
        """
        Create a new child page under the investment page with the update.
        Appends to this page when new updates come in (newest on top, older on bottom).
        Returns the child page ID.
        """
        try:
            # Notion child pages don't have a strict "newest on top" by default,
            # so we'll prefix the title with the date for sorting
            title = f"Update - {update_date}"
            
            # Build the content blocks
            children = [
                {
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": update_date, "link": None}}],
                        "color": "default",
                    }
                },
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": self._rich_text_chunks(update_content),
                        "color": "default",
                    }
                }
            ]
            
            payload = {
                "parent": {"page_id": parent_page_id},
                "properties": {
                    "title": [{"type": "text", "text": {"content": title[:2000], "link": None}}]
                },
                "children": children,
            }
            
            response = requests.post(
                f"{NOTION_API}/pages",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["id"]
        except requests.HTTPError as e:
            print(f"    HTTP Error creating child page: {e.response.status_code} {e.response.text[:200]}")
            raise
        except Exception as e:
            print(f"    Error creating child page: {e}")
            raise
    
    def append_to_child_page(
        self,
        child_page_id: str,
        update_content: str,
    ) -> None:
        """
        Append update content to an existing child page.
        New content is prepended (added at the top) so newest updates appear first.
        """
        try:
            # Get the current page content to find insertion point
            response = requests.get(
                f"{NOTION_API}/blocks/{child_page_id}/children",
                headers=self._headers(),
                timeout=15,
            )
            response.raise_for_status()
            blocks = response.json().get("results", [])
            
            # Prepend the new update by inserting at the beginning
            # We'll insert after the heading_2 (date) but before other content
            insertion_index = 1 if blocks and blocks[0].get("type") == "heading_2" else 0
            
            new_blocks = [
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": self._rich_text_chunks(update_content),
                        "color": "default",
                    }
                },
                {
                    "type": "divider",
                    "divider": {}
                }
            ]
            
            # Append blocks to the child page
            payload = {"children": new_blocks}
            
            response = requests.patch(
                f"{NOTION_API}/blocks/{child_page_id}/children",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"    Warning: Failed to append to child page: {e}")
    
    def get_or_create_child_page_for_investment(
        self,
        investment_id: str,
        investment_name: str,
    ) -> str:
        """
        Get the existing child page for updates, or create one if it doesn't exist.
        Returns the child page ID.
        """
        try:
            # Check if a child page already exists
            response = requests.get(
                f"{NOTION_API}/blocks/{investment_id}/children",
                headers=self._headers(),
                timeout=15,
            )
            response.raise_for_status()
            children = response.json().get("results", [])
            
            # Look for an existing "Updates" page
            for child in children:
                if child.get("type") == "child_page":
                    title = child.get("child_page", {}).get("title", "")
                    if "update" in title.lower():
                        return child["id"]
            
            # No existing updates page found; create one
            print(f"  Creating updates child page for {investment_name}")
            return self._create_child_page_for_investment(investment_id, investment_name)
        except Exception as e:
            print(f"    Warning: Error checking for child pages: {e}")
            # Fall through to create a new one
            return self._create_child_page_for_investment(investment_id, investment_name)
    
    def _create_child_page_for_investment(
        self,
        investment_id: str,
        investment_name: str,
    ) -> str:
        """Create a new child page for storing updates."""
        try:
            payload = {
                "parent": {"page_id": investment_id},
                "properties": {
                    "title": [{"type": "text", "text": {"content": f"{investment_name} - Updates", "link": None}}]
                },
                "children": [
                    {
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": "Investment updates (newest first)", "link": None}}],
                            "color": "default",
                        }
                    }
                ]
            }
            
            response = requests.post(
                f"{NOTION_API}/pages",
                headers=self._headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["id"]
        except Exception as e:
            print(f"    Error creating child page: {e}")
            raise
