"""Script to automatically login each website"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from playwright.sync_api import sync_playwright

REDDIT = os.environ.get("REDDIT", "http://metis.lti.cs.cmu.edu:9999")
SHOPPING = os.environ.get("SHOPPING", "https://www.amazon.ca/")
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "https://www.amazon.ca/")
GITLAB = os.environ.get("GITLAB", "gitlab.com")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "https://en.wikipedia.org/wiki/Wiki")
MAP = os.environ.get("MAP", "https://www.google.com/maps")
HOMEPAGE = os.environ.get("HOMEPAGE", "https://www.google.com/")

assert (
    REDDIT
    and SHOPPING
    and SHOPPING_ADMIN
    and GITLAB
    and WIKIPEDIA
    and MAP
    and HOMEPAGE
), (
    f"Please setup the URLs to each site. Current: \n"
    + f"Reddit: {REDDIT}\n"
    + f"Shopping: {SHOPPING}\n"
    + f"Shopping Admin: {SHOPPING_ADMIN}\n"
    + f"Gitlab: {GITLAB}\n"
    + f"Wikipedia: {WIKIPEDIA}\n"
    + f"Map: {MAP}\n"
    + f"Homepage: {HOMEPAGE}\n"
)


ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
}

URL_MAPPINGS = {
    REDDIT: "http://reddit.com",
    SHOPPING: "http://onestopmarket.com",
    SHOPPING_ADMIN: "http://luma.com/admin",
    GITLAB: "http://gitlab.com",
    WIKIPEDIA: "http://wikipedia.org",
    MAP: "http://openstreetmap.org",
    HOMEPAGE: "http://homepage.com",
}

HEADLESS = False
SLOW_MO = 0


SITES = ["gitlab", "shopping", "shopping_admin", "reddit"]
URLS = [
    f"{GITLAB}/-/profile",
    f"{SHOPPING}/wishlist/",
    f"{SHOPPING_ADMIN}/dashboard",
    f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
]
EXACT_MATCH = [True, True, True, True]
KEYWORDS = ["", "", "Dashboard", "Delete"]


def is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True
) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True

    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage"
            ],
            slow_mo=50  # 模拟真实操作，降低执行速度
    )
    context = browser.new_context(storage_state=storage_state)
    page = context.new_page()
    page.goto(url)
    time.sleep(1)
    d_url = page.url
    content = page.content()
    context_manager.__exit__()
    if keyword:
        return keyword not in content
    else:
        if url_exact:
            return d_url != url
        else:
            return url not in d_url


def renew_comb(comb: list[str], auth_folder: str = "./.auth") -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context()
    page = context.new_page()

    if "shopping" in comb:
        username = ACCOUNTS["shopping"]["username"]
        password = ACCOUNTS["shopping"]["password"]
        page.goto(f"{SHOPPING}/customer/account/login/")
        page.get_by_label("Email", exact=True).fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Sign In").click()

    if "reddit" in comb:
        username = ACCOUNTS["reddit"]["username"]
        password = ACCOUNTS["reddit"]["password"]
        print(f"Logging in to {REDDIT}, {username}, {password}")
        # also print the page content
        print(page)
        page.goto(f"{REDDIT}/login")
        print(page)
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password").fill(password)
        page.get_by_role("button", name="Log in").click()

    if "shopping_admin" in comb:
        username = ACCOUNTS["shopping_admin"]["username"]
        password = ACCOUNTS["shopping_admin"]["password"]
        page.goto(f"{SHOPPING_ADMIN}")
        page.get_by_placeholder("user name").fill(username)
        page.get_by_placeholder("password").fill(password)
        page.get_by_role("button", name="Sign in").click()

    if "gitlab" in comb:
        username = ACCOUNTS["gitlab"]["username"]
        password = ACCOUNTS["gitlab"]["password"]
        page.goto(f"{GITLAB}/users/sign_in")
        page.get_by_test_id("username-field").click()
        page.get_by_test_id("username-field").fill(username)
        page.get_by_test_id("username-field").press("Tab")
        page.get_by_test_id("password-field").fill(password)
        page.get_by_test_id("sign-in-button").click()

    context.storage_state(path=f"{auth_folder}/{'.'.join(comb)}_state.json")

    context_manager.__exit__()


def get_site_comb_from_filepath(file_path: str) -> list[str]:
    comb = os.path.basename(file_path).rsplit("_", 1)[0].split(".")
    return comb


def main(auth_folder: str = "./.auth") -> None:
    pairs = list(combinations(SITES, 2))

    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for pair in pairs:
            # TODO[shuyanzh] auth don't work on these two sites
            if "reddit" in pair and (
                "shopping" in pair or "shopping_admin" in pair
            ):
                continue
            executor.submit(
                renew_comb, list(sorted(pair)), auth_folder=auth_folder
            )

        for site in SITES:
            executor.submit(renew_comb, [site], auth_folder=auth_folder)

    futures = []
    cookie_files = list(glob.glob(f"{auth_folder}/*.json"))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for c_file in cookie_files:
            comb = get_site_comb_from_filepath(c_file)
            for cur_site in comb:
                url = URLS[SITES.index(cur_site)]
                keyword = KEYWORDS[SITES.index(cur_site)]
                match = EXACT_MATCH[SITES.index(cur_site)]
                future = executor.submit(
                    is_expired, Path(c_file), url, keyword, match
                )
                futures.append(future)

    for i, future in enumerate(futures):
        assert not future.result(), f"Cookie {cookie_files[i]} expired."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_list", nargs="+", default=["REDDIT"])
    parser.add_argument("--auth_folder", type=str, default="./.auth")
    args = parser.parse_args()
    if not args.site_list:
        main()
    else:
        if "all" in args.site_list:
            main(auth_folder=args.auth_folder)
        else:
            renew_comb(args.site_list, auth_folder=args.auth_folder)
