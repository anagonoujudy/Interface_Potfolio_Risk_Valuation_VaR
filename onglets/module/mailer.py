from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "fojumma.riskreporting@gmail.com"
SMTP_PASSWORD = "vplo rqvh sflg nqnm"
MAIL_SENDER_NAME = "FOJUMMA EQUITY"


def _validate_mail_config() -> None:
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        raise ValueError(
            "Configuration email incomplète. "
            "Veuillez définir SMTP_USERNAME et SMTP_PASSWORD dans les variables d'environnement."
        )


def send_report_email(
    recipient_email: str,
    subject: str,
    body: str,
    attachments: list[str],
) -> None:
    _validate_mail_config()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{MAIL_SENDER_NAME} <{SMTP_USERNAME}>"
    msg["To"] = recipient_email
    msg.set_content(body)

    for file_path in attachments:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        data = path.read_bytes()
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            maintype, subtype = "application", "pdf"
        elif suffix == ".xlsx":
            maintype, subtype = (
                "application",
                "vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            maintype, subtype = "application", "octet-stream"

        msg.add_attachment(
            data,
            maintype=maintype,
            subtype=subtype,
            filename=path.name,
        )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)