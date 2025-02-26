import pandas as pd
import requests
import bs4
import time
from datetime import datetime
import logging
import os

SLEEP_TIMER = 3
PATH_ABS = os.path.dirname(__file__)
PATH_LOGS = os.path.join(PATH_ABS, "logs", datetime.now().strftime("%Y-%m-%d") + ".log")
PATH_LOGS_AVOIDED = os.path.join(PATH_ABS, "logs", "cars_avoided.log")

# Configure logging
logging.basicConfig(
    filename=PATH_LOGS,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


def cars_scraper(manuf, mod):

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    manuf = manuf.replace(" ", "-").lower()
    mod = mod.replace(" ", "-").lower()
    logging.info(f"\tStarting scraping for {manuf}_{mod}...")

    initial_url = f"https://www.coches.com/coches-segunda-mano/{manuf}-{mod}.htm?page="
    
    logging.info(f"\t\tGetting number of cars listed for {manuf}_{mod}...")
    try:
        # We make a request to the first page of the search results to get the number of cars listed
        # Add headers to simulate an human user and avoid being blocked
        res = requests.get(initial_url + "0", headers=headers)
        res.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        logging.error(f"\t\t\tRequest failed for {initial_url}: {e}\n")
        return None

    soup = bs4.BeautifulSoup(res.text, "html.parser")
    
    # In every page, the number of total cars listed is shown in the subtitle of the page in the form:
    # "Tenemos <number> ofertas de <manuf> <mod> de segunda mano para esta búsqueda..."
    # We look for the number of total cars_to_get, so we can calculate the number of pages to scrape as cars_to_get/20,
    # since each page shows 20 cars. If the number of cars is less than 20, we scrape only the first page.
    # The class of the element containing the subtitle is class_to_find.
    class_to_find = "Typography-module_typography__ZANXY font-body-m SearchClassifiedSearch_description__L__Eb"
    subtitle_element = soup.find("p", class_=class_to_find)
    if subtitle_element:
        try:
            cars_to_get = int(subtitle_element.text.split(maxsplit=2)[1].replace(".", ""))
        except (IndexError, ValueError) as e:
            logging.error(f"\t\t\tFailed to parse the number of cars to get for {initial_url}: {e}")
            cars_to_get = 0
    else:
        cars_to_get = 0  # or handle this case appropriately

    links = []
    pages = int(round(cars_to_get / 20, 0))
    pages = 1 if pages < 1 else pages

    logging.info(f"\t\t\t{cars_to_get} cars and {pages} pages to get for {manuf}_{mod}.")

    if cars_to_get > 5000:
        logging.warning(f"""\t\t\tThe number of cars to get is too high ({cars_to_get})!!
 There is probably an error with the link construction due to an incorrect model name (link={initial_url+str(0)}).
 This model {mod} will be skipped and registered in the log 'cars_avoided.log'.""")
        df = None
        with open(PATH_LOGS_AVOIDED, "a") as f:
            f.write(f"{manuf}_{mod}\n")
        return df
    
    elif cars_to_get == 0:
        logging.info(f"\t\t\t0 cars listed for {manuf}_{mod}. Scraping completed for {manuf}_{mod}! 0 cars fetched.")
        df = None
        return df

    else:
        logging.info(f"\t\tScraping pages to get car ads links for {manuf}_{mod}...")
        for page in range(pages):
            logging.info(f"\t\t\tScraping page {page+1}/{pages} ({(page+1)/pages*100:.2f}% completed) for {manuf}_{mod}...")
            res = requests.get(initial_url + str(page), headers=headers)
            soup = bs4.BeautifulSoup(res.text, "html.parser")
            soup_links = soup.find_all("a", href=True)
            links += [link["href"] for link in soup_links if "/coches-segunda-mano/ocasion" in link["href"]]
            time.sleep(SLEEP_TIMER)
        logging.info(f"\t\t\tScraping of pages completed. {page+1} result pages and {len(links)} links fetched for {manuf}_{mod}.")


        table = []

        total_links = len(links)
        if total_links == 0:
            logging.info(f"\t\t\t0 links found for {manuf}_{mod}. Scraping completed for {manuf}_{mod}! 0 cars fetched.")
            df = None
            return df
        else:
            logging.info(f"\t\tScraping cars: {total_links} links for {manuf}_{mod}...")
            for link_number, link in enumerate(links):
                logging.info(f"\t\t\tScraping link {link_number+1}/{total_links} ({(link_number+1)/total_links*100:.2f}% completed) for {manuf}_{mod}...")
                # Security check in case the link is empty (program would crash). Maybe can be improved with try/except.
                if link == "":
                    logging.warning(f"\t\t\t\tLink {link_number+1} empty. Excluded from the scraping.")
                    continue

                res = requests.get(link, allow_redirects=False, headers=headers)
                if res.status_code!=200:
                    logging.warning(f"\t\t\t\tSomething went wrong with link {link_number+1}. Response {res.status_code}. Excluded from the scraping.")
                    continue
                    
                soup = bs4.BeautifulSoup(res.text, "html.parser")

                row = []
                manufacturer = manuf.replace("-", " ")
                model = mod.replace("-", " ")
                row.append(manufacturer)
                row.append(model)

                # Here we are looking for the exact version of the car, which is above the main image of the car.
                # manufacturer and model are already in the link, so we only need to find the version.
                class_to_find = "Typography-module_typography__ZANXY font-body-m text-overflow-ellipsis ClassifiedHeader_version__Ruwrt"
                version = soup.find("span", class_=class_to_find).text.strip()
                row.append(version)

                # Here we are looking for the box below the main image of the car, which contains the overview of the car,
                # with (ordered left to right and top to bottom): año, kms, combustible, cambio, potencia, puertas, color, vendedor.
                class_to_find = "FeatureText-module_wrapper__nGlYu"
                overview = soup.find_all("span", class_=class_to_find)
                # This class finds other elements that are not the overview, so we filter them out.
                filtered_overview = [item for item in overview if item.find("small")]
                
                for i, item in enumerate(filtered_overview):
                    value = item.find("span").text.strip()
                    if i==0:
                        row += value.split("/")
                    else:
                        row.append(value)

                # Here we are looking for the mini box over the overview box, which contains the two prices of the car: cash and financed.
                # However each one of them has a different class, so we need to find them separately.
                class_to_find_cash = "PriceLabel_price-label__7kJef PriceLabel_price-label-column____tr5 color-neutral-90 PriceLabel_text-align-right__vGM8u"
                class_to_find_financed = "PriceLabel_price-label__7kJef PriceLabel_price-label-column____tr5 color-secondary-principal PriceLabel_text-align-right__vGM8u"
                # Find the class_to_find, filter for the strong tag, remove trailing spaces and dots, and split the string to get the price.
                price_cash = soup.find("div", class_=class_to_find_cash).find("strong").text.strip().replace(".", "").split()[0]
                row.append(price_cash)
                # Same as above, but for the financed price.
                try:
                    price_financed = soup.find("div", class_=class_to_find_financed).find("strong").text.strip().replace(".", "").split()[0]
                except AttributeError:
                    price_financed = None
                row.append(price_financed)

                row.append(link)

                table.append(row)
                
                time.sleep(SLEEP_TIMER)
            logging.info(f"\t\t\tScraping completed for {manuf}_{mod}! {link_number+1} cars fetched.")


        df = pd.DataFrame(table, columns=(["manufacturer", "model", "version", "month", "year", "kms", "fuel", "transmission", "power_hp", "no_doors", "color",
                                        "seller", "price_cash", "price_financed", "link"]))

    return df


if __name__ == "__main__":
    pass