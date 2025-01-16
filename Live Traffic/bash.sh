#!/bin/bash

# Define los dominios que deseas consultar
DOMAINS=("example.com" "openai.com" "github.com" "cloudflare.com" "wikipedia.org")

# Endpoint DoH de Cloudflare
DOH_URL="https://cloudflare-dns.com/dns-query"

# Encabezado común para solicitudes DoH
HEADERS=(
  -H "Accept: application/dns-message"
)

# Función para generar una consulta DNS codificada en Base64
generate_dns_query() {
  local domain=$1
  # Genera una consulta DNS binaria para un registro A
  local dns_query=$(printf "\x00\x00\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00$(printf '%s' "$domain" | awk -F. '{for(i=1;i<=NF;i++)printf("%c%s",length($i),$i)}')\x00\x00\x01\x00\x01")
  # Codifica la consulta en Base64 URL-safe
  echo "$(echo -n "$dns_query" | base64 | tr -d '=' | tr '/+' '_-')"
}

# Itera sobre los dominios y realiza las consultas
for domain in "${DOMAINS[@]}"; do
  # Genera la consulta codificada
  dns_query_base64=$(generate_dns_query "$domain")
  
  # Realiza la solicitud GET
  echo "Haciendo consulta DoH a $domain con Cloudflare..."
  curl -X GET "$DOH_URL?dns=$dns_query_base64" "${HEADERS[@]}"
  echo -e "\n---\n"
done