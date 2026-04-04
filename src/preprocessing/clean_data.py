def get_severity(score):
    """
    Convert CVSS score into severity label
    """
    if score is None:
        return None
    elif score < 4:
        return "LOW"
    elif score < 7:
        return "MEDIUM"
    else:
        return "HIGH"


def clean_cve_data(cve_list):
    """
    Clean CVE data and add severity labels
    """

    cleaned = []

    for cve in cve_list:
        if cve["cvss"] is None or cve["description"] == "":
            continue

        cve["severity"] = get_severity(cve["cvss"])

        cleaned.append(cve)

    return cleaned