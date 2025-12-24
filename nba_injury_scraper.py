#!/usr/bin/env python3
"""
NBA Injury Report Scraper

Scrapes NBA injury reports from the official NBA CDN.
Supports both old format (pre-12/23/2025) and new format (12/23/2025+).

Usage:
    python nba_injury_scraper.py --season 2024-25
    python nba_injury_scraper.py --season 2022-23 --output injuries_2022.csv
    python nba_injury_scraper.py --season all
    python nba_injury_scraper.py --season 2024-25 --save-pdfs ./pdfs
    python nba_injury_scraper.py --season 2023-24 --from-pdfs ./pdfs
    python nba_injury_scraper.py --season 2023-24 --save-pdfs ./pdfs --resume
"""

import argparse
import asyncio
import aiohttp
import pandas as pd
import pdfplumber
import io
import re
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SEASON CONFIGURATION
# =============================================================================

SEASONS = {
    "2021-22": {"start": "2021-10-19", "end": "2022-04-10"},
    "2022-23": {"start": "2022-10-18", "end": "2023-04-09"},
    "2023-24": {"start": "2023-10-24", "end": "2024-04-14"},
    "2024-25": {"start": "2024-10-22", "end": "2025-04-13"},
    "2025-26": {"start": "2025-10-21", "end": "2026-04-12"},  # Projected
}

# Date when NBA switched to new URL format (with minutes)
NEW_FORMAT_START = datetime(2025, 12, 23)

BASE_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report"

# =============================================================================
# URL GENERATION
# =============================================================================

def generate_old_format_urls(date: datetime) -> list[str]:
    """
    Generate URLs for old format (hourly updates).
    Format: Injury-Report_YYYY-MM-DD_HHAM.pdf
    Hours: 12AM, 01AM, 02AM, ... 11AM, 12PM, 01PM, ... 11PM
    """
    urls = []
    date_str = date.strftime("%Y-%m-%d")
    
    # 12AM - 11AM
    for hour in range(12):
        if hour == 0:
            hour_str = "12AM"
        else:
            hour_str = f"{hour:02d}AM"
        urls.append(f"{BASE_URL}_{date_str}_{hour_str}.pdf")
    
    # 12PM - 11PM
    for hour in range(12):
        if hour == 0:
            hour_str = "12PM"
        else:
            hour_str = f"{hour:02d}PM"
        urls.append(f"{BASE_URL}_{date_str}_{hour_str}.pdf")
    
    return urls


def generate_new_format_urls(date: datetime) -> list[str]:
    """
    Generate URLs for new format (15-minute updates).
    Format: Injury-Report_YYYY-MM-DD_HH_MMAM.pdf
    """
    urls = []
    date_str = date.strftime("%Y-%m-%d")
    
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            if hour == 0:
                hour_str = "12"
                ampm = "AM"
            elif hour < 12:
                hour_str = f"{hour:02d}"
                ampm = "AM"
            elif hour == 12:
                hour_str = "12"
                ampm = "PM"
            else:
                hour_str = f"{hour - 12:02d}"
                ampm = "PM"
            
            urls.append(f"{BASE_URL}_{date_str}_{hour_str}_{minute:02d}{ampm}.pdf")
    
    return urls


def generate_urls_for_date(date: datetime) -> list[str]:
    """Generate appropriate URLs based on date."""
    if date >= NEW_FORMAT_START:
        return generate_new_format_urls(date)
    else:
        return generate_old_format_urls(date)


def get_season_dates(season: str) -> tuple[datetime, datetime]:
    """Get start and end dates for a season."""
    if season not in SEASONS:
        raise ValueError(f"Unknown season: {season}. Available: {list(SEASONS.keys())}")
    
    start = datetime.strptime(SEASONS[season]["start"], "%Y-%m-%d")
    end = datetime.strptime(SEASONS[season]["end"], "%Y-%m-%d")
    
    # Cap end date at today if in the future
    today = datetime.now()
    if end > today:
        end = today
    
    return start, end

# =============================================================================
# PDF PARSING
# =============================================================================

@dataclass
class InjuryRecord:
    report_datetime: str
    game_date: str
    game_time: str
    matchup: str
    team: str
    player_name: str
    status: str
    reason: str


def parse_injury_pdf(pdf_bytes: bytes, report_url: str) -> list[InjuryRecord]:
    """Parse injury report PDF and extract records."""
    records = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Extract report datetime from first page header
            first_page_text = pdf.pages[0].extract_text() or ""
            report_datetime = extract_report_datetime(first_page_text, report_url)

            # Combine all page text
            full_text = ""
            for page in pdf.pages:
                full_text += (page.extract_text() or "") + "\n"

            # Parse records from text
            records = parse_text_based(full_text, report_datetime)

    except Exception as e:
        logger.warning(f"Error parsing PDF {report_url}: {e}")

    return records


def parse_text_based(text: str, report_datetime: str) -> list[InjuryRecord]:
    """Parse injury records from PDF text content."""
    records = []

    # Track current game context
    current_game_date = ""
    current_game_time = ""
    current_matchup = ""
    current_team = ""

    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('Injury Report:') or line.startswith('Page'):
            continue
        if line == 'GameDate GameTime Matchup Team PlayerName CurrentStatus Reason':
            continue

        # Check for game date pattern (MM/DD/YYYY)
        date_match = re.match(r'^(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}\([A-Z]+\))\s+(\S+@\S+)\s+(.+)$', line)
        if date_match:
            current_game_date = date_match.group(1)
            current_game_time = date_match.group(2)
            current_matchup = date_match.group(3)
            remainder = date_match.group(4)
            # Parse team and player info from remainder
            record = parse_player_line(remainder, current_game_date, current_game_time,
                                      current_matchup, report_datetime)
            if record:
                current_team = record.team
                records.append(record)
            continue

        # Check for time pattern without date (same date, different game)
        time_match = re.match(r'^(\d{2}:\d{2}\([A-Z]+\))\s+(\S+@\S+)\s+(.+)$', line)
        if time_match:
            current_game_time = time_match.group(1)
            current_matchup = time_match.group(2)
            remainder = time_match.group(3)
            record = parse_player_line(remainder, current_game_date, current_game_time,
                                      current_matchup, report_datetime)
            if record:
                current_team = record.team
                records.append(record)
            continue

        # Check for matchup pattern (continuing same time slot)
        matchup_match = re.match(r'^(\S+@\S+)\s+(.+)$', line)
        if matchup_match and '@' in matchup_match.group(1):
            current_matchup = matchup_match.group(1)
            remainder = matchup_match.group(2)
            record = parse_player_line(remainder, current_game_date, current_game_time,
                                      current_matchup, report_datetime)
            if record:
                current_team = record.team
                records.append(record)
            continue

        # Check for team name starting a line (new team in same game)
        # Team names are like "AtlantaHawks", "BrooklynNets", etc.
        team_match = re.match(r'^([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)+)\s+(.+)$', line)
        if team_match and current_matchup:
            potential_team = team_match.group(1)
            # Verify it looks like an NBA team (CamelCase with multiple capitals)
            if re.match(r'^[A-Z][a-z]+[A-Z]', potential_team):
                remainder = team_match.group(2)
                record = parse_player_line_with_team(potential_team, remainder, current_game_date,
                                                     current_game_time, current_matchup, report_datetime)
                if record:
                    current_team = record.team
                    records.append(record)
                continue

        # Player line continuation (same team)
        if current_team and current_matchup:
            # Try to parse as player,name Status Reason
            player_match = re.match(r'^([A-Za-z]+,\s*[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(Out|Questionable|Doubtful|Probable|Available)\s+(.*)$', line)
            if player_match:
                record = InjuryRecord(
                    report_datetime=report_datetime,
                    game_date=current_game_date,
                    game_time=current_game_time,
                    matchup=current_matchup,
                    team=format_team_name(current_team),
                    player_name=player_match.group(1),
                    status=player_match.group(2),
                    reason=player_match.group(3)
                )
                records.append(record)

    return records


def parse_player_line(text: str, game_date: str, game_time: str, matchup: str, report_datetime: str) -> Optional[InjuryRecord]:
    """Parse a line containing team and player info."""
    # Pattern: TeamName PlayerLast,First Status Reason
    match = re.match(r'^([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)+)\s+([A-Za-z]+,\s*[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(Out|Questionable|Doubtful|Probable|Available)\s+(.*)$', text)
    if match:
        return InjuryRecord(
            report_datetime=report_datetime,
            game_date=game_date,
            game_time=game_time,
            matchup=matchup,
            team=format_team_name(match.group(1)),
            player_name=match.group(2),
            status=match.group(3),
            reason=match.group(4)
        )

    # Check for NOT YET SUBMITTED
    if 'NOTYETSUBMITTED' in text or 'NOT YET SUBMITTED' in text:
        return None

    return None


def parse_player_line_with_team(team: str, remainder: str, game_date: str, game_time: str, matchup: str, report_datetime: str) -> Optional[InjuryRecord]:
    """Parse player info when team is already extracted."""
    # Pattern: PlayerLast,First Status Reason
    match = re.match(r'^([A-Za-z]+,\s*[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(Out|Questionable|Doubtful|Probable|Available)\s+(.*)$', remainder)
    if match:
        return InjuryRecord(
            report_datetime=report_datetime,
            game_date=game_date,
            game_time=game_time,
            matchup=matchup,
            team=format_team_name(team),
            player_name=match.group(1),
            status=match.group(2),
            reason=match.group(3)
        )

    if 'NOTYETSUBMITTED' in remainder:
        return None

    return None


def format_team_name(team: str) -> str:
    """Convert CamelCase team name to readable format."""
    # Insert space before capital letters: AtlantaHawks -> Atlanta Hawks
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', team)


def extract_report_datetime(text: str, url: str) -> str:
    """Extract report datetime from PDF text or URL."""
    # Try to extract from text: "Injury Report: MM/DD/YY HH:MM AM"
    match = re.search(r'Injury Report:\s*(\d{2}/\d{2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M)', text)
    if match:
        return match.group(1)
    
    # Fall back to extracting from URL
    # Old format: Injury-Report_2024-03-17_02AM.pdf
    # New format: Injury-Report_2025-12-23_12_00AM.pdf
    match = re.search(r'Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_?(\d{2})?([AP]M)\.pdf', url)
    if match:
        date_str = match.group(1)
        hour = match.group(2)
        minute = match.group(3) or "00"
        ampm = match.group(4)
        return f"{date_str} {hour}:{minute} {ampm}"
    
    return "Unknown"


# =============================================================================
# ASYNC SCRAPING
# =============================================================================

def save_pdf(pdf_bytes: bytes, url: str, pdf_dir: str) -> None:
    """Save PDF to disk, organized by date."""
    # Extract date and filename from URL
    # e.g., Injury-Report_2024-03-17_02AM.pdf
    match = re.search(r'Injury-Report_(\d{4}-\d{2}-\d{2})_(.+\.pdf)', url)
    if match:
        date_str = match.group(1)
        filename = f"Injury-Report_{date_str}_{match.group(2)}"

        # Create date subdirectory
        date_dir = os.path.join(pdf_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)

        # Save PDF
        filepath = os.path.join(date_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(pdf_bytes)
        logger.debug(f"Saved PDF: {filepath}")


async def fetch_pdf(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    pdf_dir: Optional[str] = None,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> tuple[str, Optional[bytes]]:
    """Fetch a single PDF with rate limiting and retry logic."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        content = await response.read()
                        # Verify it's actually a PDF
                        if content[:4] == b'%PDF':
                            # Save PDF if directory specified
                            if pdf_dir:
                                save_pdf(content, url, pdf_dir)
                            return url, content
                    elif response.status == 429:  # Rate limited
                        wait_time = base_delay * (2 ** attempt)
                        logger.debug(f"Rate limited on {url}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # 404 or other error - no retry needed
                        return url, None
            except asyncio.TimeoutError:
                wait_time = base_delay * (2 ** attempt)
                logger.debug(f"Timeout on {url} (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s")
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.debug(f"Failed to fetch {url}: {e}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    return url, None
        return url, None


def check_date_downloaded(date: datetime, pdf_dir: str, min_pdfs: int = 5) -> bool:
    """Check if a date already has PDFs downloaded."""
    if not pdf_dir:
        return False
    date_str = date.strftime("%Y-%m-%d")
    date_dir = os.path.join(pdf_dir, date_str)
    if not os.path.isdir(date_dir):
        return False
    pdf_count = len([f for f in os.listdir(date_dir) if f.endswith('.pdf')])
    return pdf_count >= min_pdfs


async def scrape_date(
    session: aiohttp.ClientSession,
    date: datetime,
    semaphore: asyncio.Semaphore,
    pdf_dir: Optional[str] = None,
    resume: bool = False,
    request_delay: float = 0.1
) -> list[InjuryRecord]:
    """Scrape all injury reports for a single date."""
    # Skip if already downloaded (resume mode)
    if resume and pdf_dir and check_date_downloaded(date, pdf_dir):
        logger.debug(f"Skipping {date.date()} - already downloaded")
        # Parse from existing files instead
        date_str = date.strftime("%Y-%m-%d")
        date_dir = os.path.join(pdf_dir, date_str)
        records = []
        for filename in os.listdir(date_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(date_dir, filename)
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                parsed = parse_injury_pdf(pdf_bytes, filename)
                records.extend(parsed)
        return records

    urls = generate_urls_for_date(date)
    records = []

    # Fetch URLs with small delay between each to avoid rate limiting
    results = []
    for url in urls:
        result = await fetch_pdf(session, url, semaphore, pdf_dir)
        results.append(result)
        if request_delay > 0:
            await asyncio.sleep(request_delay)

    # Parse successful fetches
    for url, pdf_bytes in results:
        if pdf_bytes:
            parsed = parse_injury_pdf(pdf_bytes, url)
            records.extend(parsed)
            logger.debug(f"Parsed {len(parsed)} records from {url}")

    return records


async def scrape_season(
    season: str,
    max_concurrent: int = 10,
    delay_ms: int = 200,
    pdf_dir: Optional[str] = None,
    resume: bool = False,
    request_delay: float = 0.1
) -> pd.DataFrame:
    """Scrape all injury reports for a season."""
    start_date, end_date = get_season_dates(season)

    logger.info(f"Scraping season {season}: {start_date.date()} to {end_date.date()}")

    # Generate all dates
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    logger.info(f"Total days to scrape: {len(dates)}")

    # Count already downloaded if resuming
    if resume and pdf_dir:
        already_downloaded = sum(1 for d in dates if check_date_downloaded(d, pdf_dir))
        logger.info(f"Resume mode: {already_downloaded} days already downloaded, {len(dates) - already_downloaded} to fetch")

    # Calculate expected URL count
    old_format_days = sum(1 for d in dates if d < NEW_FORMAT_START)
    new_format_days = len(dates) - old_format_days
    expected_urls = old_format_days * 24 + new_format_days * 96
    logger.info(f"Expected URLs to check: ~{expected_urls}")

    if pdf_dir:
        os.makedirs(pdf_dir, exist_ok=True)
        logger.info(f"PDFs will be saved to: {pdf_dir}")

    all_records = []
    semaphore = asyncio.Semaphore(max_concurrent)

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process dates with progress bar
        for date in tqdm(dates, desc=f"Scraping {season}"):
            records = await scrape_date(session, date, semaphore, pdf_dir, resume, request_delay)
            all_records.extend(records)

            # Delay between dates to avoid rate limiting
            await asyncio.sleep(delay_ms / 1000)
    
    logger.info(f"Total records scraped: {len(all_records)}")
    
    # Convert to DataFrame
    if all_records:
        df = pd.DataFrame([vars(r) for r in all_records])
        # Remove duplicates (same player in multiple report times for same game)
        df = df.drop_duplicates(subset=['game_date', 'game_time', 'team', 'player_name', 'status', 'reason'])
        logger.info(f"Unique records after deduplication: {len(df)}")
        return df
    
    return pd.DataFrame()


def scrape_season_sync(season: str, **kwargs) -> pd.DataFrame:
    """Synchronous wrapper for scrape_season."""
    return asyncio.run(scrape_season(season, **kwargs))


def parse_local_pdfs(pdf_dir: str, season: str) -> pd.DataFrame:
    """Parse injury records from locally saved PDFs."""
    start_date, end_date = get_season_dates(season)
    logger.info(f"Parsing local PDFs for season {season}: {start_date.date()} to {end_date.date()}")

    all_records = []
    dates_processed = 0

    # Iterate through date directories
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        date_dir = os.path.join(pdf_dir, date_str)

        if os.path.isdir(date_dir):
            # Parse all PDFs in the date directory
            for filename in os.listdir(date_dir):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(date_dir, filename)
                    try:
                        with open(pdf_path, 'rb') as f:
                            pdf_bytes = f.read()
                        records = parse_injury_pdf(pdf_bytes, filename)
                        all_records.extend(records)
                        logger.debug(f"Parsed {len(records)} records from {filename}")
                    except Exception as e:
                        logger.warning(f"Error parsing {pdf_path}: {e}")
            dates_processed += 1

        current += timedelta(days=1)

    logger.info(f"Processed {dates_processed} date directories")
    logger.info(f"Total records parsed: {len(all_records)}")

    if all_records:
        df = pd.DataFrame([vars(r) for r in all_records])
        df = df.drop_duplicates(subset=['game_date', 'game_time', 'team', 'player_name', 'status', 'reason'])
        logger.info(f"Unique records after deduplication: {len(df)}")
        return df

    return pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scrape NBA injury reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python nba_injury_scraper.py --season 2024-25
    python nba_injury_scraper.py --season 2022-23 --output injuries_2022.csv
    python nba_injury_scraper.py --season all --output all_injuries.csv
    
Available seasons: """ + ", ".join(SEASONS.keys())
    )
    
    parser.add_argument(
        "--season", 
        type=str, 
        required=True,
        help="Season to scrape (e.g., 2024-25) or 'all' for all seasons"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file path (default: injuries_{season}.csv)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=500,
        help="Delay between dates in milliseconds (default: 500)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--save-pdfs",
        type=str,
        default=None,
        metavar="DIR",
        help="Save PDF files to specified directory (organized by date)"
    )
    parser.add_argument(
        "--from-pdfs",
        type=str,
        default=None,
        metavar="DIR",
        help="Parse from existing PDFs in specified directory instead of fetching from web"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip dates that already have PDFs downloaded (requires --save-pdfs)"
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.1,
        help="Delay between individual requests in seconds (default: 0.1)"
    )

    args = parser.parse_args()

    # Validate resume requires save-pdfs
    if args.resume and not args.save_pdfs:
        parser.error("--resume requires --save-pdfs to be specified")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine seasons to scrape
    if args.season.lower() == "all":
        seasons = list(SEASONS.keys())
    else:
        seasons = [args.season]
    
    # Validate seasons
    for season in seasons:
        if season not in SEASONS:
            logger.error(f"Unknown season: {season}. Available: {list(SEASONS.keys())}")
            return 1
    
    # Scrape each season
    all_dfs = []
    for season in seasons:
        if args.from_pdfs:
            # Parse from existing local PDFs
            df = parse_local_pdfs(args.from_pdfs, season)
        else:
            # Fetch from web
            df = scrape_season_sync(
                season,
                max_concurrent=args.max_concurrent,
                delay_ms=args.delay,
                pdf_dir=args.save_pdfs,
                resume=args.resume,
                request_delay=args.request_delay
            )
        if not df.empty:
            df['season'] = season
            all_dfs.append(df)
    
    # Combine results
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            season_str = args.season.replace("-", "_")
            output_path = f"injuries_{season_str}.{args.format}"
        
        # Save
        if args.format == "parquet":
            combined_df.to_parquet(output_path, index=False)
        else:
            combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(combined_df)} records to {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"Total records: {len(combined_df)}")
        print(f"Date range: {combined_df['game_date'].min()} to {combined_df['game_date'].max()}")
        print(f"Unique players: {combined_df['player_name'].nunique()}")
        print(f"Unique teams: {combined_df['team'].nunique()}")
        print(f"Output file: {output_path}")
        
        # Status breakdown
        print(f"\nStatus breakdown:")
        for status, count in combined_df['status'].value_counts().items():
            print(f"  {status}: {count}")
    else:
        logger.warning("No records found!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
