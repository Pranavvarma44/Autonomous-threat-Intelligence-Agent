import json


def load_cve_data(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cve_list = []

    # JSON 2.0 structure
    for item in data.get("vulnerabilities", []):

        cve = item.get("cve", {})

        # CVE ID
        cve_id = cve.get("id", "")

        # Description
        descriptions = cve.get("descriptions", [])
        description = ""

        for d in descriptions:
            if d.get("lang") == "en":
                description = d.get("value")
                break

        # Initialize all fields
        cvss = None
        exploitability = None
        impact_score = None
        confidentiality = None
        integrity = None
        availability = None

        # Extract metrics safely
        try:
            metrics = cve.get("metrics", {})
            cvss_metrics = metrics.get("cvssMetricV31", [])

            if len(cvss_metrics) > 0:
                metric = cvss_metrics[0]

                cvss_data = metric.get("cvssData", {})

                # Core score
                cvss = cvss_data.get("baseScore")

                # Additional scores
                exploitability = metric.get("exploitabilityScore")
                impact_score = metric.get("impactScore")

                # CIA impacts
                confidentiality = cvss_data.get("confidentialityImpact")
                integrity = cvss_data.get("integrityImpact")
                availability = cvss_data.get("availabilityImpact")

        except Exception as e:
            pass  # keep values as None if anything fails

        cve_list.append({
            "cve_id": cve_id,
            "description": description,
            "cvss": cvss,
            "exploitability": exploitability,
            "impact_score": impact_score,
            "confidentiality": confidentiality,
            "integrity": integrity,
            "availability": availability
        })

    return cve_list