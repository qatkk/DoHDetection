#To execute it -> bash iptablesrules.sh


#Create a blacklist of blocking ips
ipset create blacklist hash:ip
#Loop that reads from /kali/home/Desktop/file, where each line contains an IP that we want to block
while read line
do
ipset add blacklist "$line" #Add the IP to block into the blacklist
done < /kali/home/Desktop/file
#Finally it lasts to block the src and dst packets which IP matches any of the specified in the blacklist
iptables -I forwarding_lan_rule -m set --match-set blacklist src -j drop
iptables -I forwarding_lan_rule -m set --match-set blacklist dst -j drop