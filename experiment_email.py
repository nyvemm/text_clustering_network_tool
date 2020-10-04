import ssl
import smtplib
import stats_analyser

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def sendEmail(sender, basename, path):
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "sendmail@gmail.com"
    receiver_email = "receivermail@gmail.com"
    password = "somepassword"
    
    stats = stats_analyser.postprocessing(path)
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "{}: Finished Processing [{}]".format(sender, basename)
    msg['From'] = sender_email
    msg['To'] = receiver_email

    text = """<fieldset>
   <h1 style="font-family: sans-serif;font-weight:lighter">Finished Processing a database</h1>
   <h2 style="font-family: sans-serif;font-weight:lighter;line-height:5px;"><b>Machine</b>: {0}</h2>
   <br>
   <h2 style="font-family: sans-serif;font-weight:lighter;line-height:5px;"><b>Basename</b>: {1}</h2>
   <h2 style="font-family: sans-serif;font-weight:lighter;line-height:5px;"><b>Path</b>: <code style="color:blue">{3}</code></h2>
   <br>
   <hr>
   <h1 style="font-family: sans-serif;font-weight:lighter"> Results:</h1>
   <pre>{2}</pre>
</fieldset>""".format(sender, basename, stats, path)

    html = MIMEText(text, 'html')
    msg.attach(html)

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
