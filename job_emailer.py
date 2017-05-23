import os
import socket
import smtplib
import random
import argparse

from email.mime.text import MIMEText

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('to', help='destination email address')
    args = parser.parse_args()

    machine_name = socket.gethostname().split('.')[0]
    session_name = os.environ['STY'].split('.')[1]

    max_valid = 0

    #### Dummy data for now >>>
    for i in range(5):
        r = random.random() * 100
        if r > max_valid:
            max_valid = r
    steps = random.randint(1,1000)
    #### <<<

    # Create a file called .env with the following contents:
    ''' 
    USERNAME:<username@gmail.com>
    PASSWORD:<password>
    '''
    USERNAME = 0
    PASSWORD = 1

    fp = open('.env', 'r')
    ENV = fp.read()
    ENV_TOKENS = ENV.split('\n')
    username = ENV_TOKENS[USERNAME].split(':')[1]
    password = ENV_TOKENS[PASSWORD].split(':')[1]
    fp.close()

    msg = MIMEText('Ran for %d steps \n\n modify body as needed' % steps)

    msg['Subject'] = '%s @ %s finished with %.4f' % (session_name, machine_name, max_valid)
    msg['From'] = username
    msg['To'] = args.to

    # Send the message via GMAIL SMTP server, but don't include the
    # envelope header.
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(username, password)
    server.sendmail(username, [args.to], msg.as_string())
    server.quit()